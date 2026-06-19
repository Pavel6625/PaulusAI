"""Run the digital person as a simple terminal chat.

  paulus            # after `pip install`
  python -m paulus  # equivalent

Commands:
  /sleep    run consolidation (distil facts, propose skills, decay memory)
  /mood     show current mood
  /memory   show the inspectable semantic memory
  /skills   list learned skills
  /quit     consolidate and exit
"""
from . import affect, agent, config, memory, skills, vectorstore


def main():
    config.ensure_dirs()
    vectorstore.init()  # bring up embeddings, or fall back to keyword search
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

        print(f"\ndp> {agent.respond(text)}\n")


if __name__ == "__main__":
    main()
