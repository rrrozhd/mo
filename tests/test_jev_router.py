import json
import pytest

from experiments import jev_router as router


class FakeResponse:
    """Stands in for the Jev endpoint; synthetic probabilities are never a real decision."""

    def __init__(self, payload):
        self._payload = json.dumps(payload).encode()

    def read(self):
        return self._payload

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False


def answer(choice, confidence):
    other = "chemistry" if choice == "batch" else "batch"
    return {"model": "jev-latest", "answers": {"route": {
        "type": "choice", "choice": choice, "confidence": confidence,
        "probabilities": {choice: confidence, other: round(1 - confidence, 3)}}}}


def test_a_confident_choice_is_routed(monkeypatch):
    monkeypatch.setattr(router, "jev_key", lambda *a, **k: "test-key")
    monkeypatch.setattr(router, "urlopen", lambda *a, **k: FakeResponse(answer("chemistry", 0.97)))
    decision = router.route("Is there a cleaner analog?")
    assert decision["agent"] == "chemistry" and decision["status"] == "routed"


def test_low_confidence_between_two_specialists_still_answers(monkeypatch):
    """Low confidence used to mean "ask". It now means "either could answer", which is
    different from "this is not our question" — that is what the off-topic class is for."""
    monkeypatch.setattr(router, "jev_key", lambda *a, **k: "test-key")
    monkeypatch.setattr(router, "urlopen", lambda *a, **k: FakeResponse(answer("chemistry", 0.55)))
    decision = router.route("Which of those would you take forward?")
    assert decision["status"] == "routed_either" and decision["agent"] == "chemistry"


def test_an_unreachable_router_falls_back_to_the_batch_assistant(monkeypatch):
    monkeypatch.setattr(router, "jev_key", lambda *a, **k: "test-key")

    def explode(*_args, **_kwargs):
        raise OSError("connection refused")

    monkeypatch.setattr(router, "urlopen", explode)
    decision = router.route("Require candidate-20")
    assert decision["agent"] == "batch" and decision["status"] == "router_unavailable"


def test_an_unusable_answer_is_refused(monkeypatch):
    monkeypatch.setattr(router, "jev_key", lambda *a, **k: "test-key")
    monkeypatch.setattr(router, "urlopen", lambda *a, **k: FakeResponse({"answers": {}}))
    with pytest.raises(ValueError, match="no usable choice"):
        router.classify("anything")
    with pytest.raises(ValueError, match="nonempty"):
        router.classify("   ")


@pytest.mark.parametrize("payload", [[], None, {"answers": None},
    {"answers": {"route": {"choice": [], "confidence": 1}}},
    {"answers": {"route": {"choice": {}, "confidence": 1}}}] + [
    {"answers": {"route": {"choice": "chemistry", "confidence": confidence}}}
    for confidence in [None, True, 4, -1, float("inf"), float("nan")]])
def test_malformed_router_output_uses_unavailable_fallback(monkeypatch, payload):
    monkeypatch.setattr(router, "jev_key", lambda: "test-key")
    monkeypatch.setattr(router, "urlopen", lambda *a, **kw: FakeResponse(payload))
    decision = router.route("Inspect a")
    assert decision["agent"] == "batch"
    assert decision["status"] == "router_unavailable"


def test_context_is_folded_into_the_state_but_the_last_message_is_the_subject():
    state = router._state("and candidate-12?", [
        {"message": "Does candidate-08 inhibit CYP3A4?",
         "answer": {"explanation": "Yes, 0.76."}}])
    assert state.endswith("Message to classify:\nand candidate-12?")
    assert "Chemist: Does candidate-08 inhibit CYP3A4?" in state
    assert "Assistant: Yes, 0.76." in state


def test_context_is_bounded_and_empty_context_sends_the_message_alone():
    assert router._state("hello", None) == "hello"
    assert router._state("hello", []) == "hello"
    assert router._state("hello", [{"message": "", "answer": None}]) == "hello"
    state = router._state("now", [{"message": "x" * 5000, "answer": {"explanation": "y" * 5000}}])
    assert len(state) < 2 * router.MAX_CONTEXT_CHARS + 200


def test_context_reaches_the_classifier(monkeypatch):
    seen = {}
    monkeypatch.setattr(router, "jev_key", lambda *a, **k: "test-key")

    def capture(request, timeout=None):
        seen["state"] = json.loads(request.data)["state"]
        return FakeResponse(answer("chemistry", 0.95))

    monkeypatch.setattr(router, "urlopen", capture)
    router.route("and that one?", context=[{"message": "Profile candidate-08",
                                            "answer": {"explanation": "Score 0.76."}}])
    assert "Profile candidate-08" in seen["state"]
    assert seen["state"].endswith("and that one?")


def test_an_on_topic_message_both_specialists_could_answer_is_still_answered(monkeypatch):
    monkeypatch.setattr(router, "jev_key", lambda *a, **k: "test-key")
    payload = {"model": "jev", "answers": {"route": {
        "type": "choice", "choice": "chemistry", "confidence": 0.56,
        "probabilities": {"batch": 0.29, "chemistry": 0.71, "other": 0.0}}}}
    monkeypatch.setattr(router, "urlopen", lambda *a, **k: FakeResponse(payload))
    decision = router.route("Which of those would you take forward?")
    assert decision["status"] == "routed_either" and decision["agent"] == "chemistry"


def test_an_off_topic_message_clarifies_even_when_confident(monkeypatch):
    monkeypatch.setattr(router, "jev_key", lambda *a, **k: "test-key")
    payload = {"model": "jev", "answers": {"route": {
        "type": "choice", "choice": "other", "confidence": 1.0,
        "probabilities": {"batch": 0.0, "chemistry": 0.0, "other": 1.0}}}}
    monkeypatch.setattr(router, "urlopen", lambda *a, **k: FakeResponse(payload))
    assert router.route("What is the weather?")["status"] == "unclear"


def test_a_confident_specialist_with_off_topic_mass_still_clarifies(monkeypatch):
    monkeypatch.setattr(router, "jev_key", lambda *a, **k: "test-key")
    payload = {"model": "jev", "answers": {"route": {
        "type": "choice", "choice": "chemistry", "confidence": 0.72,
        "probabilities": {"batch": 0.0, "chemistry": 0.72, "other": 0.28}}}}
    monkeypatch.setattr(router, "urlopen", lambda *a, **k: FakeResponse(payload))
    assert router.route("Half a question")["status"] == "unclear"
