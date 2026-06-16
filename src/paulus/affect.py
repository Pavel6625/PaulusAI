"""Persistent mood, driven by the OCC appraisal engine.

Mood is a PAD vector (Pleasure, Arousal, Dominance), each in [-1, 1] — the
standard dimensional representation that discrete emotions map onto. Each
appraised emotion nudges the mood toward that emotion's PAD signature; mood then
decays back toward a personality-determined baseline.

Still functional, still legible: describe() reports the mood and the last
emotion felt, and feel() returns the emotions so the caller can log *why*.
This is expressive, explainable emotional behaviour — not a claim of feeling.
"""
import json

from . import appraisal, config

# Stable traits (Big Five subset). 0..1.
PERSONALITY = {
    "extraversion": 0.6,   # raises pleasure baseline / sociability
    "neuroticism": 0.4,    # amplifies negative emotions, lowers baseline
    "reactivity": 0.7,     # overall emotional gain
}

# PAD signatures for each OCC emotion (approximate Mehrabian-style mappings).
EMOTION_PAD = {
    "joy":            (0.40, 0.20, 0.10),
    "distress":       (-0.40, -0.20, -0.30),
    "hope":           (0.20, 0.20, -0.10),
    "fear":           (-0.60, 0.50, -0.60),
    "satisfaction":   (0.30, -0.10, 0.20),
    "fears_confirmed": (-0.50, 0.10, -0.40),
    "disappointment": (-0.30, -0.10, -0.20),
    "relief":         (0.30, -0.20, 0.10),
    "pride":          (0.40, 0.30, 0.30),
    "shame":          (-0.30, 0.10, -0.40),
    "admiration":     (0.30, 0.10, -0.10),
    "reproach":       (-0.30, 0.10, 0.10),
    "gratitude":      (0.40, 0.20, -0.20),
    "anger":          (-0.50, 0.60, 0.30),
    "gratification":  (0.50, 0.30, 0.30),
    "remorse":        (-0.40, 0.10, -0.30),
}

_DIMS = ("pleasure", "arousal", "dominance")


def _baseline():
    p = 0.2 * (PERSONALITY["extraversion"] - PERSONALITY["neuroticism"])
    return {"pleasure": round(p, 3), "arousal": 0.0, "dominance": 0.0}


def _load():
    if config.AFFECT_FILE.exists():
        return json.loads(config.AFFECT_FILE.read_text(encoding="utf-8"))
    s = _baseline()
    s["last_emotion"] = None
    return s


def _save(s):
    config.AFFECT_FILE.write_text(json.dumps(s, indent=2), encoding="utf-8")


def _clip(x):
    return max(-1.0, min(1.0, x))


def feel(event):
    """Appraise an event, fold the resulting emotions into the mood, persist,
    and return the {emotion: intensity} dict (for logging the 'why')."""
    emotions = appraisal.appraise_event(
        event, PERSONALITY["reactivity"], PERSONALITY["neuroticism"]
    )
    if not emotions:
        return {}
    s = _load()
    gain = 0.5
    for emo, inten in emotions.items():
        pad = EMOTION_PAD.get(emo)
        if not pad:
            continue
        for i, dim in enumerate(_DIMS):
            s[dim] = round(_clip(s[dim] + pad[i] * inten * gain), 3)
    s["last_emotion"] = appraisal.dominant(emotions)
    _save(s)
    return emotions


def decay(s=None):
    """Mood drifts back toward the personality baseline each turn."""
    s = s or _load()
    base = _baseline()
    for dim in _DIMS:
        s[dim] = round(base[dim] + (s[dim] - base[dim]) * 0.85, 3)
    _save(s)
    return s


def describe(s=None):
    s = s or _load()
    p, a = s["pleasure"], s["arousal"]
    tone = "positive" if p > 0.2 else "low" if p < -0.2 else "neutral"
    energy = "high-energy" if a > 0.4 else "calm" if a < 0.1 else "alert"
    felt = f", last felt {s['last_emotion']}" if s.get("last_emotion") else ""
    return (f"{tone}, {energy}{felt} "
            f"(P={p:+.2f} A={a:+.2f} D={s['dominance']:+.2f})")
