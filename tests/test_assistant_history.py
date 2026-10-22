from src.workbench import assistant as assistant


def turn(message, reply, status="complete"):
    return {"message": message, "status": status, "answer": {"explanation": reply}}


def test_history_replays_completed_exchanges_oldest_first():
    messages = assistant._history([turn("first", "one"), turn("second", "two")])
    assert [m["role"] for m in messages] == ["user", "assistant", "user", "assistant"]
    assert [m["content"] for m in messages] == ["first", "one", "second", "two"]


def test_history_skips_turns_that_never_produced_an_answer():
    assert assistant._history([turn("running", "", status="running"),
                               turn("failed", None, status="failed"),
                               {"message": "no answer key", "status": "complete"}]) == []


def test_history_is_bounded_in_turns_and_length():
    many = [turn(f"message {index}", "x" * 4000) for index in range(20)]
    messages = assistant._history(many)
    assert len(messages) == 2 * assistant.MAX_HISTORY_TURNS
    assert messages[0]["content"] == "message 14"  # Oldest kept, not oldest overall.
    assert all(len(m["content"]) <= assistant.MAX_HISTORY_CHARS for m in messages)


def test_no_history_is_an_empty_conversation():
    assert assistant._history(None) == [] and assistant._history([]) == []
