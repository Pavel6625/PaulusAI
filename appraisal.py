"""Appraisal engine (OCC model, compact).

Replaces the old hand-tuned "event -> valence delta" table with a principled
pipeline: an event is described by appraisal *variables* (how desirable it is
relative to goals, how praiseworthy relative to standards, who caused it, and
whether it's actual/prospective/confirmed), and the engine derives discrete
emotions (joy, fear, pride, gratitude, ...) with intensities from those
variables. affect.py then folds those emotions into a persistent mood.

This is the Ortony-Clore-Collins structure, trimmed to the branches an MVP
companion actually exercises. It stays legible: every emotion can be traced
back to the appraisal that produced it.
"""
from dataclasses import dataclass

NEGATIVE = {
    "distress", "fear", "fears_confirmed", "disappointment",
    "shame", "reproach", "anger", "remorse",
}


@dataclass
class Appraisal:
    desirability: float = 0.0       # -1..1, relative to the agent's goals
    praiseworthiness: float = 0.0   # -1..1, relative to standards
    agency: str = "none"            # "self" | "other" | "none"
    prospect: str = "actual"        # actual | prospective | confirmed | disconfirmed
    likelihood: float = 1.0         # 0..1, for prospective events
    unexpectedness: float = 0.0     # 0..1, boosts intensity


def appraise(a: Appraisal, reactivity=0.7, neuroticism=0.4):
    """Return {emotion: intensity} for a single appraised event."""
    emo = {}
    d, p = a.desirability, a.praiseworthiness

    # --- Event vs. goals (well-being + prospect) ---------------------------
    if a.prospect == "actual":
        if d > 0:
            emo["joy"] = d
        elif d < 0:
            emo["distress"] = -d
    elif a.prospect == "prospective":
        (emo.__setitem__("hope", d * a.likelihood) if d > 0
         else emo.__setitem__("fear", -d * a.likelihood) if d < 0 else None)
    elif a.prospect == "confirmed":
        emo["satisfaction" if d > 0 else "fears_confirmed"] = abs(d)
    elif a.prospect == "disconfirmed":
        emo["disappointment" if d > 0 else "relief"] = abs(d)

    # --- Action vs. standards (attribution) --------------------------------
    if p != 0 and a.agency != "none":
        if a.agency == "self":
            emo["pride" if p > 0 else "shame"] = abs(p)
        else:
            emo["admiration" if p > 0 else "reproach"] = abs(p)

    # --- Compound emotions (well-being + attribution, actual events) -------
    if a.prospect == "actual" and a.agency != "none" and p != 0:
        if a.agency == "self":
            if d > 0 and p > 0:
                emo["gratification"] = (d + p) / 2
            if d < 0 and p < 0:
                emo["remorse"] = (-d - p) / 2
        else:
            if d > 0 and p > 0:
                emo["gratitude"] = (d + p) / 2
            if d < 0 and p < 0:
                emo["anger"] = (-d - p) / 2

    # --- Intensity scaling: reactivity, surprise, neuroticism bias ---------
    boost = 1 + 0.5 * a.unexpectedness
    out = {}
    for k, v in emo.items():
        scale = reactivity * boost
        if k in NEGATIVE:
            scale *= (0.7 + 0.6 * neuroticism)
        out[k] = round(min(1.0, max(0.0, v * scale)), 3)
    return {k: v for k, v in out.items() if v > 0.01}


# Library of appraisal templates for the coarse events the agent emits.
# This is the bridge from simple agent signals to full OCC processing.
EVENT_APPRAISALS = {
    "owner_thanks":     Appraisal(desirability=0.7, praiseworthiness=0.6, agency="self"),
    "owner_frustrated": Appraisal(desirability=-0.6, praiseworthiness=-0.5, agency="self"),
    "task_success":     Appraisal(desirability=0.5, praiseworthiness=0.4, agency="self"),
    "task_error":       Appraisal(desirability=-0.5, praiseworthiness=-0.4, agency="self"),
    "action_declined":  Appraisal(desirability=-0.2, agency="none"),
    "new_learning":     Appraisal(desirability=0.4, prospect="prospective", likelihood=0.8),
}


def appraise_event(event, reactivity=0.7, neuroticism=0.4):
    a = EVENT_APPRAISALS.get(event)
    if a is None:
        return {}
    return appraise(a, reactivity, neuroticism)


def dominant(emotions):
    if not emotions:
        return None
    return max(emotions.items(), key=lambda kv: kv[1])[0]
