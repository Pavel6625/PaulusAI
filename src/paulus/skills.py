"""Procedural memory: a skill library, inspired by Hermes Agent's
learn-from-experience loop.

A skill is a named, reusable procedure the agent distils from a successful
interaction. New skills start as 'unverified' (proposed by the consolidation
loop) — the design doc's reflection-review gate: an unverified skill is a
*suggestion*, and any action it implies still passes the per-action safety gate.
A skill becomes 'verified' once it's been used successfully.
"""
import datetime
import json
import re

from . import config

_WORD = re.compile(r"[a-z0-9]+")


def _load():
    if config.SKILLS_FILE.exists():
        return json.loads(config.SKILLS_FILE.read_text(encoding="utf-8"))
    return []


def _save(skills):
    config.SKILLS_FILE.write_text(json.dumps(skills, indent=2), encoding="utf-8")


def add_skill(name, when_to_use, steps, status="unverified", source="experience"):
    skills = _load()
    for s in skills:
        if s["name"].strip().lower() == name.strip().lower():
            s.update(when_to_use=when_to_use, steps=steps)
            _save(skills)
            return f"updated skill '{name}'"
    skills.append({
        "name": name,
        "when_to_use": when_to_use,
        "steps": steps,
        "status": status,
        "uses": 0,
        "created": datetime.datetime.now().isoformat(timespec="seconds"),
        "source": source,
    })
    _save(skills)
    return f"saved {status} skill '{name}'"


def mark_used(name, success=True):
    skills = _load()
    for s in skills:
        if s["name"].strip().lower() == name.strip().lower():
            s["uses"] += 1
            if success and s["status"] == "unverified":
                s["status"] = "verified"
            _save(skills)
            return
    return


def describe() -> str:
    """A human-readable listing of every skill and its status. Used by the
    /skills command on the CLI and the chat gateway."""
    skills = _load()
    if not skills:
        return "(no skills yet)"
    return "\n".join(
        f"- [{s['status']}] {s['name']} (uses {s['uses']}): {s['when_to_use']}"
        for s in skills
    )


def find_skills(query, k=3):
    skills = _load()
    q = set(_WORD.findall(query.lower()))
    scored = []
    for s in skills:
        hay = set(_WORD.findall((s["name"] + " " + s["when_to_use"]).lower()))
        overlap = len(q & hay)
        if overlap:
            scored.append((overlap, s))
    scored.sort(key=lambda x: -x[0])
    return [s for _, s in scored[:k]]
