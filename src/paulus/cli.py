"""Run the digital person as a simple terminal chat.

  paulus            # after `pip install`
  python -m paulus  # equivalent

Commands:
  /sleep    run consolidation (distil facts, propose skills, decay memory)
  /mood     show current mood
  /memory   show the inspectable semantic memory
  /skills   list learned skills
  /auto     toggle auto mode: a reviewer model approves high-impact actions
            you clearly asked for; anything else still prompts you
  /bg X     run X as a background task; its result is printed when done
  /tasks    list background tasks
  /cancel N cancel background task N
  /mcp      list connected MCP servers and their tools
  /quit     consolidate and exit
"""
from . import affect, agent, config, mcp_client, memory, skills, tasks, vectorstore


def main():
    config.ensure_dirs()
    vectorstore.init()  # bring up embeddings, or fall back to keyword search
    mcp_client.init()   # connect configured MCP servers (none = no MCP tools)
    print("PaulusAI. Type a message, or /quit to exit.")
    print(f"permission mode: {config.PERMISSION_MODE}  (/auto to toggle)\n")
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
        if text == "/auto":
            config.PERMISSION_MODE = "ask" if config.PERMISSION_MODE == "auto" else "auto"
            print(f"permission mode: {config.PERMISSION_MODE}")
            continue
        if text == "/bg" or text.startswith("/bg "):
            print(tasks.start_from_owner(text[len("/bg"):]))
            continue
        if text == "/tasks":
            print(tasks.status(None))
            continue
        if text == "/cancel" or text.startswith("/cancel "):
            print(tasks.cancel(None, text[len("/cancel"):].strip()))
            continue
        if text == "/mcp":
            print(mcp_client.status())
            continue

        print(f"\ndp> {agent.respond(text)}\n")


if __name__ == "__main__":
    main()
