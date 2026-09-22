import hashlib
import json
import sys
from types import SimpleNamespace

import pytest


def test_sessions_export_md_writes_single_session(monkeypatch, tmp_path, capsys):
    import hermes_cli.main as main_mod
    import hermes_state

    captured = {}

    class FakeDB:
        def resolve_session_id(self, session_id):
            captured["resolved_from"] = session_id
            return "20260706_123456_abcd1234"

        def export_session(self, session_id, *, include_compacted=False):
            captured["exported"] = session_id
            return {
                "id": session_id,
                "title": "Export CLI Test",
                "source": "cli",
                "message_count": 1,
                "messages": [{"role": "user", "content": "hello"}],
            }

        def delete_session(self, *args, **kwargs):
            raise AssertionError("markdown export must not delete sessions")

        def prune_sessions(self, *args, **kwargs):
            raise AssertionError("markdown export must not prune sessions")

        def close(self):
            captured["closed"] = True

    monkeypatch.setattr(hermes_state, "SessionDB", lambda *args, **kwargs: FakeDB())
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "hermes",
            "sessions",
            "export",
            "--format",
            "md",
            "--session-id",
            "20260706_123456",
            str(tmp_path),
        ],
    )

    main_mod.main()

    output = capsys.readouterr().out
    files = list(tmp_path.glob("*.md"))
    assert len(files) == 1
    text = files[0].read_text(encoding="utf-8")
    assert "# Export CLI Test" in text
    assert "hello" in text
    assert captured == {
        "resolved_from": "20260706_123456",
        "exported": "20260706_123456_abcd1234",
        "closed": True,
    }
    assert "Exported 1 session" in output
    assert "1 message" in output
    assert str(files[0]) in output


def test_sessions_export_redact_scrubs_secrets(monkeypatch, tmp_path):
    """--redact runs exported content through force-mode secret redaction."""
    import hermes_cli.main as main_mod
    import hermes_state

    secret = "sk-proj-Zz12345678901234567890123456789012345678"

    class FakeDB:
        def resolve_session_id(self, session_id):
            return "s1"

        def export_session(self, session_id, *, include_compacted=False):
            return {
                "id": "s1",
                "title": "Redact",
                "messages": [
                    {"role": "tool", "name": "terminal", "content": f"api key: {secret}"}
                ],
            }

        def close(self):
            pass

    monkeypatch.setattr(hermes_state, "SessionDB", lambda *args, **kwargs: FakeDB())
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "hermes", "sessions", "export", "--format", "md",
            "--session-id", "s1", "--redact", str(tmp_path),
        ],
    )

    main_mod.main()

    text = next(tmp_path.glob("*.md")).read_text(encoding="utf-8")
    assert secret not in text
    assert "api key:" in text


def _seed_compacted_export_session(db, session_id, **session_fields):
    db.create_session(session_id, source=session_fields.pop("source", "cli"), **session_fields)
    original = [
        {"role": "user", "content": f"{session_id}: original request"},
        {"role": "assistant", "content": f"{session_id}: original answer"},
        {"role": "user", "content": f"{session_id}: carried request"},
        {"role": "assistant", "content": f"{session_id}: carried answer"},
    ]
    db.append_messages_batch(session_id, original)
    tail = db.get_messages(session_id)[-2:]
    summary = {"role": "assistant", "content": f"{session_id}: compact summary"}
    # Older compaction generations retain both copies of the protected tail.
    db.archive_and_compact(session_id, [summary, *tail])
    removed = db.append_message(session_id, "user", f"{session_id}: undone request")
    db.append_message(session_id, "assistant", f"{session_id}: undone answer")
    db.rewind_to_message(session_id, removed)
    db.end_session(session_id, "cli_close")
    return [message["content"] for message in original] + [summary["content"]]


def _run_transcript_export(output_dir, session_id, fmt, lineage="single", delete=False):
    from hermes_cli.sessions_cmd import cmd_sessions

    return cmd_sessions(SimpleNamespace(
        sessions_action="export", session_id=session_id, format=fmt,
        output=str(output_dir), redact=False, only=None, force=False,
        delete_after_verified=delete, yes=delete, lineage=lineage, dry_run=False,
    ))


def _assert_exported_transcript(output_dir, session_id, fmt, contents, lineage_ids):
    from hermes_cli.session_export_md import verify_export_file

    path = next(output_dir.glob(f"{session_id}-*.{fmt}"))
    text = path.read_text(encoding="utf-8")
    for content in contents:
        assert text.count(content) == 1, f"Export must retain exactly one copy of {content!r}"
    positions = [text.index(content) for content in contents]
    assert positions == sorted(positions)
    assert "undone request" not in text
    assert "undone answer" not in text
    assert f"- Exported messages: `{len(contents)}`" in text
    expected = {"id": session_id, "messages": [{"content": content} for content in contents]}
    assert verify_export_file(path, expected) == (True, "ok")
    records = [json.loads(line) for line in (output_dir / "manifest.jsonl").read_text().splitlines()]
    entry = next(record for record in records if record["session_id"] == session_id)
    assert entry["message_count"] == len(contents)
    assert entry["lineage_session_ids"] == lineage_ids
    assert entry["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    return text


@pytest.mark.parametrize("fmt", ["md", "qmd"])
@pytest.mark.parametrize("lineage", ["single", "logical"])
@pytest.mark.parametrize("delete", [False, True])
def test_readable_export_preserves_compacted_history(tmp_path, fmt, lineage, delete):
    """A verified archive retains original turns, but not duplicate generations or undone work."""
    from hermes_state import SessionDB

    db = SessionDB()
    try:
        db.create_session("ancestor", source="cli")
        db.append_message("ancestor", "user", "earlier compression segment")
        db.end_session("ancestor", "compression")
        contents = _seed_compacted_export_session(db, "conversation", parent_session_id="ancestor")
        active = db.get_messages("conversation")
        # Resume/import consumers keep their active-context default projection.
        assert db.export_session("conversation")["messages"] == active
        assert db.export_session_lineage("conversation")["segments"][-1]["messages"] == active
        assert next(row for row in db.export_all() if row["id"] == "conversation")["messages"] == active
        stored_count = db.message_count("conversation")
    finally:
        db.close()

    output_dir = tmp_path / "exports"
    _run_transcript_export(output_dir, "conversation", fmt, lineage, delete)
    expected = (["earlier compression segment"] if lineage == "logical" else []) + contents
    lineage_ids = ["ancestor", "conversation"] if lineage == "logical" else ["conversation"]
    text = _assert_exported_transcript(output_dir, "conversation", fmt, expected, lineage_ids)
    if lineage == "single":
        assert "earlier compression segment" not in text
    db = SessionDB()
    try:
        assert (db.get_session("conversation") is None) == delete
        assert db.message_count("conversation") == (0 if delete else stored_count)
        assert db.get_session("ancestor") is not None
    finally:
        db.close()


@pytest.mark.parametrize("fmt", ["md", "qmd"])
def test_verified_export_preserves_compacted_delegate_history(tmp_path, fmt):
    """Cascade deletion is permitted only after the parent's and delegate's original turns are archived."""
    from hermes_state import SessionDB

    db = SessionDB()
    try:
        parent_contents = _seed_compacted_export_session(db, "parent")
        delegate_contents = _seed_compacted_export_session(
            db, "delegate", source="delegate", parent_session_id="parent",
            model_config={"_delegate_from": "parent"},
        )
    finally:
        db.close()

    output_dir = tmp_path / "exports"
    _run_transcript_export(output_dir, "parent", fmt, delete=True)
    _assert_exported_transcript(output_dir, "parent", fmt, parent_contents, ["parent"])
    _assert_exported_transcript(output_dir, "delegate", fmt, delegate_contents, ["delegate"])
    db = SessionDB()
    try:
        assert db.get_session("parent") is None
        assert db.get_session("delegate") is None
        assert db.message_count() == 0
    finally:
        db.close()


def _trace_fake_db(captured):
    class FakeDB:
        def resolve_session_id(self, session_id):
            return "s1"

        def get_session(self, session_id):
            return {"id": session_id, "model": "test-model"}

        def get_messages_as_conversation(self, session_id):
            captured["conv"] = session_id
            return [
                {"role": "user", "content": "hello trace"},
                {"role": "assistant", "content": "hi"},
            ]

        def close(self):
            captured["closed"] = True

    return FakeDB()




