"""Task-scoped grants: the owner's "allow similar actions for this task".

Approving every step of a multi-step job one tap at a time is the friction
auto mode was meant to remove, yet a reviewer judging each step on its own
keeps flagging the routine follow-ups. A grant is the owner's own answer to
that: approving an action with "allow for this task" lets later actions of the
same kind in the same task run without asking again.

A grant is deliberately narrow and short-lived:
  - it lives on the task's ReviewContext (automode.py), so it ends with the
    turn or background task it was given in, and is never stored;
  - it covers one kind of action: writing workspace files, running one
    program (or one subcommand of a multiplexer like git), or sending to one
    recipient. Never "anything";
  - some actions can't be granted at all: deleting, killing, network clients,
    and wrappers that run other programs (sudo, sh -c, xargs, ...). The owner
    sees those one at a time, always;
  - a covered command must stay inside the workspace. An absolute, home,
    parent or $-expanded path, command substitution or a multi-line script
    sends it back to the normal review/prompt.
"""
import os
import re
import shlex

# Programs a grant never covers. Wrappers run some other program, so a grant
# for them would cover anything; the rest are destructive, or reach machines.
_NEVER = frozenset({
    # wrappers / interpreters of whole command lines
    "sh", "bash", "zsh", "dash", "fish", "ksh", "env", "xargs", "eval", "exec",
    "nohup", "timeout", "nice", "time", "command", "builtin", "source", ".",
    "sudo", "su", "doas", "watch", "parallel", "find",
    # destructive or hard to undo
    "rm", "rmdir", "dd", "shred", "truncate", "mkfs", "chmod", "chown", "chgrp",
    "kill", "pkill", "killall", "shutdown", "reboot", "halt", "poweroff",
    "systemctl", "service", "crontab", "mount", "umount",
    # reach other machines
    "curl", "wget", "ssh", "scp", "sftp", "rsync", "nc", "ncat", "netcat",
    "socat", "telnet", "ftp",
})
# Multiplexers whose first argument is the real action: a grant for
# `git status` must not cover `git push`.
_SUBCOMMANDS = frozenset({
    "git", "pip", "pip3", "npm", "pnpm", "yarn", "uv", "cargo", "go", "docker",
    "apt", "apt-get", "brew", "poetry", "conda",
})
_SEPARATORS = frozenset({";", "&&", "||", "|", "&", "|&"})
_PROGRAM = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_.+-]*$")
_ASSIGNMENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")
_SUBSTITUTION = re.compile(r"`|\$\(|<\(|>\(")
_SAFE_ABSOLUTE = frozenset({"/dev/null", "/dev/stdout", "/dev/stderr"})
# A URL argument isn't a path (but file:// is).
_URL = re.compile(r"^(?!file:)[A-Za-z][A-Za-z0-9+.-]*://[^\s]*$", re.IGNORECASE)


def keys(tool_name, tool_input):
    """The grant keys this action needs, as a frozenset, or None if no grant
    can ever cover it. Granting an action adds its keys; a later action is
    covered when all of its keys have been granted."""
    if not isinstance(tool_input, dict):
        return None
    if tool_name == "write_local_file":
        # The sandbox already confines the path to the owner's workspace.
        return frozenset({"write_local_file"})
    if tool_name == "run_command":
        return _command_keys(tool_input.get("command"))
    if tool_name in ("send_message", "send_document", "send_email_agentmail"):
        to = str(tool_input.get("to") or "").strip().lower()
        if not to and tool_name != "send_document":
            return None
        return frozenset({f"{tool_name}:{to}"})
    return None


def covers(granted, tool_name, tool_input):
    needed = keys(tool_name, tool_input)
    return bool(needed) and needed <= granted


def describe(needed):
    """What granting *needed* allows, in words for the approval prompt."""
    parts = []
    programs = sorted(k.split(":", 1)[1] for k in needed if k.startswith("run_command:"))
    if programs:
        parts.append("running " + ", ".join(f"`{p}`" for p in programs))
    if "write_local_file" in needed:
        parts.append("writing files in the workspace")
    for k in sorted(needed):
        tool, _, to = k.partition(":")
        if tool == "send_message":
            parts.append(f"messages to {to}")
        elif tool == "send_email_agentmail":
            parts.append(f"emails to {to}")
        elif tool == "send_document":
            parts.append(f"documents to {to or 'this chat'}")
    return "; ".join(parts)


def _command_keys(command):
    if not isinstance(command, str) or not command.strip():
        return None
    # A newline is a second command that shlex would read as more arguments to
    # the first; substitution runs a program the key never names.
    if "\n" in command or "\r" in command or _SUBSTITUTION.search(command):
        return None
    try:
        lexer = shlex.shlex(command, posix=True, punctuation_chars=True)
        lexer.whitespace_split = True
        tokens = list(lexer)
    except ValueError:                       # unbalanced quotes
        return None

    found = set()
    segment = []
    for token in tokens + [";"]:
        if token in _SEPARATORS:
            if segment:
                key = _segment_key(segment)
                if key is None:
                    return None
                found.add(key)
            segment = []
        else:
            segment.append(token)
    return frozenset(found) or None


def _segment_key(tokens):
    # Assignments are checked too: LD_PRELOAD=/tmp/x.so must not ride along.
    if not all(_inside_workspace(w) for w in tokens):
        return None
    words = list(tokens)
    while words and _ASSIGNMENT.match(words[0]):
        words.pop(0)                         # FOO=bar program ...
    if not words:
        return None
    program = os.path.basename(words[0])
    if not _PROGRAM.match(program) or program in _NEVER:
        return None
    if program in _SUBCOMMANDS:
        sub = next((w for w in words[1:] if not w.startswith("-")), "")
        if sub and not _PROGRAM.match(sub):
            return None
        return f"run_command:{program} {sub}".rstrip()
    return f"run_command:{program}"


def _inside_workspace(word):
    """False for any word that could point outside the workspace."""
    if word in _SAFE_ABSOLUTE:
        return True
    if "$" in word or word.startswith(("~", "/")):
        return False
    if _URL.match(word):
        return True
    if "://" in word:
        return False
    # Redirections arrive as their own tokens (">", ">>", "2>&1" pieces), and a
    # path glued to an option (--out=/etc/x) still has to be checked.
    for piece in re.split(r"[=,:]", word):
        if piece.startswith(("/", "~")) and piece not in _SAFE_ABSOLUTE:
            return False
        if ".." in piece.replace("\\", "/").split("/"):
            return False
    return True
