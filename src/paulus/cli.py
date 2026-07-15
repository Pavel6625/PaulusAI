"""Run the digital person as a simple terminal chat.

  paulus            # after `pip install`
  python -m paulus  # equivalent

Commands:
  /sleep    run consolidation (distil facts, propose skills, decay memory)
  /mood     show current mood
  /memory   show the inspectable semantic memory
  /skills   list learned skills
  /route X  show which model tier the text X would be routed to, and why
  /routes   show recent routing decisions, their outcomes, and what was learned
  /quit     consolidate and exit
"""
from . import affect, agent, config, memory, router, skills, vectorstore


def _routes_report(user_id=None, limit=15):
    """Recent decisions with the outcome each one drew, so an under-route is
    visible as a row: routed low, then had to reach outside its own memory."""
    log = router.read_log(user_id)
    if not log:
        return "(no routing decisions logged yet)"
    flags = {}
    for r in log:
        if r.get("type") == "outcome":
            flags.setdefault(r["ref"], []).append(
                "effort:" + ",".join(r.get("effort_tools", [])))
        elif r.get("type") == "complaint":
            flags.setdefault(r["ref"], []).append("complained")

    decisions = [r for r in log if r.get("type") == "decision"][-limit:]
    lines = [f"{'tier':<5} {'text':<40} {'why':<34} outcome"]
    for d in decisions:
        flat = (d.get("text") or "").replace("\n", " ")[:38]
        why = (d.get("reason") or "")[:32]
        note = "; ".join(flags.get(d["id"], [])) or "-"
        lines.append(f"{d['tier']:<5} {flat:<40} {why:<34} {note}")

    lines.append("")
    lines.append("learned exemplars:")
    lines.append(router.learned_summary(user_id))
    if not config.ROUTE_LEARNING:
        lines.append("  (DP_ROUTE_LEARNING is off — nothing is promoted)")
    return "\n".join(lines)


def main():
    config.ensure_dirs()
    vectorstore.init()  # bring up embeddings, or fall back to keyword search
    router.init()       # bring up routing, or pin every turn to the core model
    print("PaulusAI. Type a message, or /quit to exit.\n")
    while True:
        try:
            text = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            text = "/quit"

        if not text:
            continue
        if text == "/quit":
            print("\n" + agent.sleep())
            print("goodbye.")
            break
        if text == "/sleep":
            print(agent.sleep())
            continue
        if text == "/mood":
            print("mood:", affect.describe())
            continue
        if text == "/memory":
            print(memory.semantic_text())
            continue
        if text == "/skills":
            print(skills.describe())
            continue
        if text.startswith("/route "):
            probe = text[len("/route "):].strip()
            model, tier, reason = router.route(probe)
            print(f"tier: {tier}  model: {model}\nwhy:  {reason}")
            continue
        if text == "/routes":
            print(_routes_report())
            continue

        print(f"\ndp> {agent.respond(text)}\n")


if __name__ == "__main__":
    main()
