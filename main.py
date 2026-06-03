"""Run the digital person as a simple terminal chat.

  export ANTHROPIC_API_KEY=sk-...
  python main.py

Commands:
  /sleep    run consolidation (distil facts, propose skills, decay memory)
  /mood     show current mood
  /memory   show the inspectable semantic memory
  /skills   list learned skills
  /quit     consolidate and exit
"""
import affect
import agent
import config
import skills
import vectorstore


def main():
    vectorstore.init()  # bring up embeddings, or fall back to keyword search
    print("Digital Person MVP. Type a message, or /quit to exit.\n")
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
            print(agent.sleep()); continue
        if text == "/mood":
            print("mood:", affect.describe()); continue
        if text == "/memory":
            print(config.SEMANTIC_MD.read_text(encoding="utf-8")
                  if config.SEMANTIC_MD.exists() else "(no semantic memory yet)")
            continue
        if text == "/skills":
            sk = skills._load()
            if not sk:
                print("(no skills yet)")
            for s in sk:
                print(f"- [{s['status']}] {s['name']} (uses {s['uses']}): {s['when_to_use']}")
            continue

        print(f"\ndp> {agent.respond(text)}\n")


if __name__ == "__main__":
    main()
