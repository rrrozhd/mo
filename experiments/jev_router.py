"""Route one chat message to the specialist that should answer it.

This isolated routing experiment compares two scopes: the batch assistant, which may revise the
selection, and the chemistry analyst, which may not. Sending a message to the wrong
one is worse than asking, so a routing decision below the confidence floor becomes a
clarification rather than a guess.

Jev is a typed-decision model: it returns the chosen label with a calibrated
confidence instead of prose, which is the whole reason it is used here rather than
another chat model pretending to be a classifier.
"""

import json
from http.client import HTTPException
import os
from pathlib import Path
import shlex
from urllib.error import URLError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parent.parent
API = "https://api.typesafe.ai/v1/systemone"
MODEL = "jev-latest"
TIMEOUT = 10
# Off-topic messages land near 0.6 in testing, so the floor sits above that.
CONFIDENCE_FLOOR = 0.7
OFF_TOPIC = "other"
# A specialist may win narrowly because either could answer. That is not the same as a
# message this product has no answer for, so the classifier is asked to say which it is.
MAX_OFF_TOPIC = 0.25
CRITERIA = {
    "batch": "Changing the compound selection: the compound budget, the diversity or "
             "similarity limit, requiring or excluding a candidate, marking the batch "
             "reviewed, exporting the session, or asking what the current batch is.",
    "chemistry": "A question about molecules themselves: predicted CYP inhibition for a "
                 "structure, finding a safer or cleaner analog, published assay evidence, "
                 "comparing two structures, or why a molecule scores as it does.",
    OFF_TOPIC: "Anything else: small talk, a question about the weather or the news, or a "
               "request this CYP screening tool has no way to answer.",
}


def jev_key(root=ROOT):
    key = os.environ.get("JEV_API_KEY")
    if key:
        return key
    # Read one trusted local assignment as data; never source/execute the file.
    for line in (root / ".env").read_text().splitlines():
        name, separator, value = line.partition("=")
        if separator and name.strip() == "JEV_API_KEY":
            fields = shlex.split(value, comments=True)
            if len(fields) == 1:
                return fields[0]
    raise ValueError("No JEV_API_KEY in the environment or .env")


MAX_CONTEXT_TURNS = 4
MAX_CONTEXT_CHARS = 600


def _state(message, context):
    """What Jev reads. A follow-up like "and candidate-12?" routes on the exchange, not itself."""
    if not context:
        return message
    lines = []
    for turn in context[-MAX_CONTEXT_TURNS:]:
        said = (turn.get("message") or "").strip()
        reply = ((turn.get("answer") or {}).get("explanation") or "").strip()
        if said:
            lines.append(f"Chemist: {said[:MAX_CONTEXT_CHARS]}")
        if reply:
            lines.append(f"Assistant: {reply[:MAX_CONTEXT_CHARS]}")
    if not lines:
        return message
    return ("Earlier in this conversation:\n" + "\n".join(lines)
            + f"\n\nMessage to classify:\n{message}")


def classify(message, criteria=None, timeout=TIMEOUT, context=None):
    """Ask Jev which specialist should handle the message. Raises on any transport failure."""
    if not isinstance(message, str) or not message.strip():
        raise ValueError("message must be a nonempty string")
    criteria = criteria or CRITERIA
    payload = json.dumps({
        "model": MODEL,
        "state": _state(message, context),
        "questions": {"route": {
            "type": "choice",
            "instructions": "Which specialist should handle the chemist's final message? "
                            "Earlier exchanges are context for resolving what it refers to; "
                            "classify the final message only.",
            "criteria": criteria,
        }},
    }).encode()
    request = Request(API, data=payload, method="POST", headers={
        "Authorization": f"Bearer {jev_key()}",
        "Content-Type": "application/json",
    })
    with urlopen(request, timeout=timeout) as response:
        body = json.loads(response.read())
    answers = body.get("answers") if isinstance(body, dict) else None
    answer = answers.get("route") if isinstance(answers, dict) else None
    if (not isinstance(answer, dict) or not isinstance(answer.get("choice"), str)
            or answer["choice"] not in criteria):
        raise ValueError(f"Router returned no usable choice: {body}")
    confidence = answer.get("confidence")
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)) or not 0 <= confidence <= 1:
        raise ValueError("Router confidence must be a finite number in [0, 1]")
    return {
        "agent": answer["choice"],
        "confidence": float(confidence),
        "probabilities": answer.get("probabilities", {}),
        "model": body.get("model", MODEL),
    }


def route(message, fallback="batch", floor=CONFIDENCE_FLOOR, context=None):
    """Decide who answers, degrading safely rather than guessing.

    A transport failure falls back to the batch assistant, which is what handled every
    message before routing existed. A confident-enough choice is used. Anything else is
    returned as 'unclear' so the caller can ask instead of spending on the wrong agent.
    """
    try:
        decision = classify(message, context=context)
    except (HTTPException, URLError, ValueError, TimeoutError, OSError) as exc:
        return {"agent": fallback, "confidence": None, "probabilities": {},
                "status": "router_unavailable", "detail": f"{type(exc).__name__}: {exc}"}
    probabilities = decision["probabilities"] or {}
    off_topic = float(probabilities.get(OFF_TOPIC, 0.0))
    if decision["agent"] == OFF_TOPIC or off_topic > MAX_OFF_TOPIC:
        return {**decision, "status": "unclear", "floor": floor}
    if decision["confidence"] >= floor:
        return {**decision, "status": "routed", "floor": floor}
    # Both specialists are plausible and the message is on topic, so answering beats asking.
    return {**decision, "status": "routed_either", "floor": floor}
