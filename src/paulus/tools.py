"""The agent's 'body': tools with typed schemas and narrow scope.

File and command execution route through a pluggable Sandbox (sandbox.py), so
isolation is a config choice (local / Docker / SSH), not baked in. High-impact
tools are declared in security.HIGH_IMPACT_TOOLS and gated by the agent loop.
"""
from . import config, memory, sandbox, security, skills

_sbx = sandbox.get_sandbox()

TOOL_SPECS = [
    {
        "name": "remember",
        "description": "Store a durable fact about the owner or world in long-term semantic memory.",
        "input_schema": {
            "type": "object",
            "properties": {
                "fact": {"type": "string", "description": "A concise, standalone fact."},
                "confidence": {"type": "number"},
            },
            "required": ["fact"],
        },
    },
    {
        "name": "recall",
        "description": "Semantically search long-term memory for facts relevant to a query.",
        "input_schema": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    },
    {
        "name": "find_skill",
        "description": "Look up a learned procedure relevant to the current task.",
        "input_schema": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    },
    {
        "name": "save_skill",
        "description": "Save a reusable procedure learned from a successful task.",
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "when_to_use": {"type": "string"},
                "steps": {"type": "string"},
            },
            "required": ["name", "when_to_use", "steps"],
        },
    },
    {
        "name": "read_local_file",
        "description": "Read a text file from the sandboxed workspace. Returns its content as UNTRUSTED data.",
        "input_schema": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
    },
    {
        "name": "write_local_file",
        "description": "Write a text file into the sandboxed workspace. HIGH-IMPACT: requires owner confirmation.",
        "input_schema": {
            "type": "object",
            "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
            "required": ["path", "content"],
        },
    },
    {
        "name": "run_command",
        "description": "Run a shell command in the sandbox backend. HIGH-IMPACT: requires owner confirmation.",
        "input_schema": {
            "type": "object",
            "properties": {"command": {"type": "string"}},
            "required": ["command"],
        },
    },
    {
        "name": "send_message",
        "description": "Send a message on the owner's behalf. HIGH-IMPACT: requires owner confirmation.",
        "input_schema": {
            "type": "object",
            "properties": {"to": {"type": "string"}, "body": {"type": "string"}},
            "required": ["to", "body"],
        },
    },
    {
        "name": "send_document",
        "description": "Send a text document as a file attachment (e.g. notes, a report, code). "
                       "HIGH-IMPACT: requires owner confirmation.",
        "input_schema": {
            "type": "object",
            "properties": {
                "to": {"type": "string", "description": "Optional. platform:chat_id, or bare chat_id. "
                                                        "Omit to send to the current conversation."},
                "filename": {"type": "string", "description": "The attachment's file name, e.g. notes.md."},
                "content": {"type": "string", "description": "The document's text content."},
            },
            "required": ["filename", "content"],
        },
    },
    {
        "name": "list_emails_agentmail",
        "description": "List or search for messages in the AgentMail inbox.",
        "input_schema": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "Optional search query."}},
        },
    },
    {
        "name": "send_email_agentmail",
        "description": "Send an email using the AgentMail service.",
        "input_schema": {
            "type": "object",
            "properties": {
                "to": {"type": "string"},
                "subject": {"type": "string"},
                "body": {"type": "string"},
            },
            "required": ["to", "subject", "body"],
        },
    },
]


def execute(name, tool_input, user_id=None):
    """Run a tool. Returns (result_text, is_error). Any required confirmation
    for high-impact tools is obtained by the caller BEFORE this runs."""
    try:
        if name == "remember":
            res = memory.add_fact(tool_input["fact"],
                                  confidence=float(tool_input.get("confidence", 0.7)),
                                  user_id=user_id)
            security.audit("remember", tool_input["fact"])
            return res, False

        if name == "recall":
            hits = memory.search_facts(tool_input["query"], user_id=user_id)
            if not hits:
                return "No relevant facts found.", False
            return "\n".join(f"- {h['fact']} (salience {h['salience']:.2f})" for h in hits), False

        if name == "find_skill":
            hits = skills.find_skills(tool_input["query"])
            if not hits:
                return "No matching skill. Solve it directly, then consider save_skill.", False
            return "\n".join(
                f"- [{s['status']}] {s['name']}: {s['when_to_use']}\n  steps: {s['steps']}"
                for s in hits), False

        if name == "save_skill":
            res = skills.add_skill(tool_input["name"], tool_input["when_to_use"],
                                   tool_input["steps"], status="verified", source="agent")
            security.audit("save_skill", tool_input["name"])
            return res, False

        if name == "read_local_file":
            content = _sbx.read_file(tool_input["path"], user_id=user_id)
            security.audit("read_local_file", tool_input["path"])
            return security.wrap_untrusted(f"file:{tool_input['path']}", content), False

        if name == "write_local_file":
            res = _sbx.write_file(tool_input["path"], tool_input["content"], user_id=user_id)
            security.audit("write_local_file", tool_input["path"])
            return res, False

        if name == "run_command":
            out = _sbx.run(tool_input["command"], user_id=user_id)
            security.audit("run_command", f"[{config.SANDBOX_BACKEND}] {tool_input['command']}")
            return security.wrap_untrusted("command_output", out), False

        if name == "send_message":
            security.audit("send_message", f"to={tool_input['to']}")
            from .gateway.runner import get_runner
            runner = get_runner()
            if runner is not None:
                result = runner.dispatch_outbound(tool_input["to"], tool_input["body"])
            else:
                result = f"[simulated] message sent to {tool_input['to']}."
            return result, False

        if name == "send_document":
            to = tool_input.get("to", "")
            security.audit("send_document",
                           f"to={to or '(current chat)'} file={tool_input['filename']}")
            from .gateway.runner import get_runner
            runner = get_runner()
            if runner is not None:
                result = runner.dispatch_document(
                    to, tool_input["filename"], tool_input["content"], user_id=user_id
                )
            else:
                result = f"[simulated] document {tool_input['filename']} sent."
            return result, False

        if name == "list_emails_agentmail":
            import os
            from agentmail import AgentMail
            client = AgentMail(api_key=os.environ.get("AGENTMAIL_API_KEY"))
            emails = client.list_emails(query=tool_input.get("query"))
            return str(emails), False

        if name == "send_email_agentmail":
            import os
            from agentmail import AgentMail
            client = AgentMail(api_key=os.environ.get("AGENTMAIL_API_KEY"))
            res = client.send_email(
                to=tool_input["to"], 
                subject=tool_input["subject"], 
                body=tool_input["body"]
            )
            security.audit("send_email_agentmail", f"to={tool_input['to']}")
            return str(res), False

        return f"Unknown tool: {name}", True
    except Exception as e:
        return f"Tool error: {e}", True
