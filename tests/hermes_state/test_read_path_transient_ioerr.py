"""#100871: a transient SQLITE_IOERR on a warm pooled read connection is retried on the same
connection, then surfaced -- never quarantined, never a close+reopen."""
import sqlite3

import pytest

import hermes_state
from hermes_state import SessionDB


_STATE = {"failures_left": 0, "attempts": 0}  # module-level: the tracking factory subclasses _FlakyReads


class _FlakyReads(sqlite3.Connection):
    """Real SQLite connection whose first N SELECTs fail the way a mid-checkpoint mode=ro reader does."""

    def execute(self, sql, *args, **kwargs):  # type: ignore[override]
        if str(sql).lstrip().upper().startswith("SELECT"):
            _STATE["attempts"] += 1
            if _STATE["failures_left"] > 0:
                _STATE["failures_left"] -= 1
                raise sqlite3.OperationalError("disk I/O error")
        return super().execute(sql, *args, **kwargs)


@pytest.fixture
def db(tmp_path, monkeypatch):
    monkeypatch.setattr(hermes_state, "_READ_ONLY_IOERR_RETRY_BACKOFF_S", 0.0)
    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session("s", "cli")
    if not db._wal_active:
        db.close()
        pytest.skip("read pool needs WAL")
    real_connect = hermes_state._connect_tracked_db

    def flaky_connect(path, *args, **kwargs):
        if str(path).startswith("file:"):
            kwargs["factory"] = _FlakyReads
        return real_connect(path, *args, **kwargs)

    monkeypatch.setattr(hermes_state, "_connect_tracked_db", flaky_connect)
    while db._evict_one_idle_read_conn():  # the next read opens through the flaky factory
        pass
    _STATE.update(failures_left=0, attempts=0)
    yield db
    db.close()


def test_transient_ioerr_on_pooled_read_is_retried_on_the_same_connection(db):
    _STATE["failures_left"] = 1
    row = db.get_session("s")
    assert row is not None and row["id"] == "s"
    assert _STATE["attempts"] == 2  # one failure, one replay
    assert db._db_corrupt is False and db._db_wal_generation_lost is False


def test_persistent_ioerr_propagates_after_the_budget(db):
    _STATE["failures_left"] = 10 ** 6
    with pytest.raises(sqlite3.OperationalError, match="disk I/O error"):
        db.get_session("s")
    assert _STATE["attempts"] == hermes_state._READ_ONLY_IOERR_RETRY_ATTEMPTS + 1
    assert db._db_corrupt is False  # busy/EIO is not corruption: no quarantine


@pytest.fixture
def display_db(tmp_path, monkeypatch):
    """Inject page-read failures on real connections, in either journal mode."""
    state = {"remaining": 0, "attempts": 0, "stage": "execute", "connections": []}

    def maybe_fail(stage):
        if state["stage"] == stage and state["remaining"]:
            state["remaining"] -= 1
            raise sqlite3.OperationalError(state.get("error", "disk I/O error"))

    class PageCursor(sqlite3.Cursor):
        def fetchall(self):
            maybe_fail("fetch")
            return super().fetchall()

    class PageConnection(sqlite3.Connection):
        def execute(self, sql, *args, **kwargs):
            if str(sql).lstrip().upper().startswith("WITH PAGE AS"):
                state["attempts"] += 1
                state["connections"].append(self)
                maybe_fail("execute")
                return self.cursor(factory=PageCursor).execute(sql, *args, **kwargs)
            return super().execute(sql, *args, **kwargs)

    real_connect = hermes_state._connect_tracked_db

    def connect(*args, **kwargs):
        kwargs["factory"] = PageConnection
        return real_connect(*args, **kwargs)

    monkeypatch.setattr(hermes_state, "_connect_tracked_db", connect)
    monkeypatch.setattr(hermes_state, "_READ_ONLY_IOERR_RETRY_BACKOFF_S", 0.0)
    db = SessionDB(tmp_path / "display.db")
    try:
        db.create_session("s", "cli")
        db.append_messages_batch("s", [
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": "answer"},
        ])
        yield db, state
    finally:
        db.close()


@pytest.mark.parametrize("stage", ["execute", "fetch"])
def test_display_page_retries_transient_ioerr(display_db, stage):
    db, state = display_db
    expected = db.get_messages("s")
    state.update(remaining=1, stage=stage)

    assert db.get_messages("s", include_compacted=True) == expected
    assert state["attempts"] == 2
    assert state["connections"][0] is state["connections"][1]
    assert all(not conn.in_transaction for conn in state["connections"])
    assert db._db_corrupt is False


@pytest.mark.parametrize("error", ["disk I/O error", "database disk image is malformed"])
def test_display_page_failure_budget_and_transaction_cleanup(display_db, error):
    db, state = display_db
    budget = hermes_state._READ_ONLY_IOERR_RETRY_ATTEMPTS + 1
    state.update(remaining=budget + 1, error=error)

    with pytest.raises(sqlite3.OperationalError, match=error):
        db.get_messages("s", include_compacted=True)

    assert state["attempts"] == (budget if error == "disk I/O error" else 1)
    assert all(not conn.in_transaction for conn in state["connections"])
    state["remaining"] = 0
    assert len(db.get_messages("s", include_compacted=True)) == 2
