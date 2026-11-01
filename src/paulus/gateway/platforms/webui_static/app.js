"use strict";

// PaulusAI web chat. Talks to the gateway's web adapter (webui.py): REST for
// the session and history, one WebSocket for everything else.

const $ = (id) => document.getElementById(id);
const log = $("log");
const input = $("input");
const fileInput = $("file");

const IMAGE_MAX_SIDE = 2048;          // larger images are downscaled before upload
const PLATFORM_NAMES = { telegram: "Telegram", web: "web" };

let ws = null;
let retryDelay = 1000;
let streaming = null;                 // { el, text } of the reply being streamed
let renderQueued = false;
let attachments = [];                 // { name, type, data } ready to send
const approvals = new Map();          // approval id -> card element

// --- Rendering --------------------------------------------------------------

function renderMarkdown(el, text) {
  if (window.marked && window.DOMPurify) {
    el.innerHTML = DOMPurify.sanitize(marked.parse(text, { breaks: true, gfm: true }));
    for (const a of el.querySelectorAll("a")) {
      a.target = "_blank";
      a.rel = "noopener noreferrer";
    }
  } else {
    el.style.whiteSpace = "pre-wrap";
    el.textContent = text;
  }
}

function nearBottom() {
  return log.scrollHeight - log.scrollTop - log.clientHeight < 120;
}

function append(el) {
  const stick = nearBottom();
  log.append(el);
  if (stick) log.scrollTop = log.scrollHeight;
  return el;
}

function ownerBubble(text, files = [], via = null) {
  const el = document.createElement("div");
  el.className = "msg owner";
  el.textContent = text;
  if (files.length) {
    const f = document.createElement("span");
    f.className = "files";
    f.textContent = files.map((n) => "📎 " + n).join("  ");
    el.append(f);
  }
  if (via && via !== "web") {
    const m = document.createElement("span");
    m.className = "meta";
    m.textContent = "via " + (PLATFORM_NAMES[via] || via);
    el.append(m);
  }
  return append(el);
}

function agentBubble(text) {
  const el = document.createElement("div");
  el.className = "msg agent";
  renderMarkdown(el, text);
  return append(el);
}

function systemNote(text) {
  const el = document.createElement("div");
  el.className = "msg system";
  el.textContent = text;
  return append(el);
}

function safeUrl(url) {
  try {
    const u = new URL(url, location.href);
    return u.protocol === "https:" || u.protocol === "http:" ? u.href : null;
  } catch {
    return null;
  }
}

// --- Streaming --------------------------------------------------------------

function streamDelta(text) {
  if (!streaming) {
    const el = document.createElement("div");
    el.className = "msg agent streaming";
    streaming = { el: append(el), text: "" };
  }
  streaming.text += text;
  if (!renderQueued) {
    renderQueued = true;
    requestAnimationFrame(() => {
      renderQueued = false;
      if (!streaming) return;
      const stick = nearBottom();
      renderMarkdown(streaming.el, streaming.text);
      if (stick) log.scrollTop = log.scrollHeight;
    });
  }
}

function streamCommit(text) {
  if (streaming) {
    const stick = nearBottom();
    streaming.el.classList.remove("streaming");
    renderMarkdown(streaming.el, text);
    if (stick) log.scrollTop = log.scrollHeight;
    streaming = null;
  } else {
    agentBubble(text);
  }
}

function streamDiscard() {
  if (streaming) streaming.el.remove();
  streaming = null;
}

// --- Approvals --------------------------------------------------------------

function approvalCard(ev) {
  if (approvals.has(ev.id)) return;          // replayed on reconnect
  const card = document.createElement("div");
  card.className = "approval";
  const pre = document.createElement("pre");
  pre.textContent = ev.prompt;
  const actions = document.createElement("div");
  actions.className = "actions";
  const choices = [["ok", "✅ Approve", "btn primary"], ["no", "🚫 Deny", "btn"]];
  if (ev.grantable) choices.push(["all", "✅ Allow for this task", "btn"]);
  for (const [answer, label, cls] of choices) {
    const b = document.createElement("button");
    b.type = "button";
    b.className = cls;
    b.textContent = label;
    b.addEventListener("click", () => {
      for (const x of actions.querySelectorAll("button")) x.disabled = true;
      send({ type: "approve", id: ev.id, answer });
    });
    actions.append(b);
  }
  card.append(pre, actions);
  approvals.set(ev.id, card);
  append(card);
}

function approvalSettled(ev) {
  const card = approvals.get(ev.id);
  if (!card) return;
  card.querySelector(".actions")?.remove();
  const v = document.createElement("div");
  v.className = "verdict";
  v.textContent = ev.verdict;
  card.append(v);
}

// --- Socket -----------------------------------------------------------------

function setStatus(busy) {
  $("status").hidden = !busy;
}

function setConn(state) {
  const c = $("conn");
  c.className = "conn " + state;
  c.title = state === "on" ? "Connected" : "Reconnecting…";
}

function send(event) {
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify(event));
    return true;
  }
  systemNote("Not connected — reconnecting, please try again in a moment.");
  return false;
}

function onEvent(ev) {
  switch (ev.type) {
    case "user":
      ownerBubble(ev.text, ev.attachments || []);
      break;
    case "typing":
      setStatus(true);
      break;
    case "idle":
      setStatus(false);
      break;
    case "delta":
      streamDelta(ev.text);
      break;
    case "segment":
    case "final":
      streamCommit(ev.text);
      break;
    case "discard":
      streamDiscard();
      break;
    case "message": {
      const el = agentBubble(ev.text || "");
      const url = ev.link && safeUrl(ev.link.url);
      if (url) {
        const a = document.createElement("a");
        a.className = "btn primary";
        a.href = url;
        a.target = "_blank";
        a.rel = "noopener noreferrer";
        a.textContent = ev.link.label || "Open";
        el.append(a);
      }
      break;
    }
    case "document": {
      const el = agentBubble("");
      const a = document.createElement("a");
      a.className = "btn";
      a.href = URL.createObjectURL(new Blob([ev.content], { type: "text/plain;charset=utf-8" }));
      a.download = ev.filename;
      a.textContent = "📄 " + ev.filename;
      el.append(a);
      break;
    }
    case "approval":
      approvalCard(ev);
      break;
    case "approval_settled":
      approvalSettled(ev);
      break;
  }
}

function connect() {
  const url = new URL("ws", location.href);
  url.protocol = location.protocol === "https:" ? "wss:" : "ws:";
  ws = new WebSocket(url);
  ws.onopen = () => {
    retryDelay = 1000;
    setConn("on");
  };
  ws.onmessage = (m) => {
    try {
      onEvent(JSON.parse(m.data));
    } catch (e) {
      console.error(e);
    }
  };
  ws.onclose = async () => {
    setConn("off");
    streamDiscard();
    setStatus(false);
    // A closed socket may mean the session ended; don't spin on a 403.
    const me = await fetch("api/me", { credentials: "same-origin" }).catch(() => null);
    if (me && me.status === 401) return showLogin(await me.json());
    setTimeout(connect, retryDelay);
    retryDelay = Math.min(retryDelay * 2, 30000);
  };
}

// --- Composer ---------------------------------------------------------------

function autosize() {
  input.style.height = "auto";
  input.style.height = Math.min(input.scrollHeight, 200) + "px";
}

function renderChips() {
  const box = $("attachments");
  box.replaceChildren();
  attachments.forEach((att, i) => {
    const chip = document.createElement("span");
    chip.className = "chip";
    const name = document.createElement("span");
    name.textContent = "📎 " + att.name;
    const x = document.createElement("button");
    x.type = "button";
    x.textContent = "✕";
    x.title = "Remove";
    x.addEventListener("click", () => {
      attachments.splice(i, 1);
      renderChips();
    });
    chip.append(name, x);
    box.append(chip);
  });
}

function readAsBase64(blob) {
  return new Promise((resolve, reject) => {
    const r = new FileReader();
    r.onload = () => resolve(String(r.result).split(",", 2)[1] || "");
    r.onerror = () => reject(r.error);
    r.readAsDataURL(blob);
  });
}

async function shrinkImage(file) {
  // Big photos cost tokens and bandwidth; the model gains little past ~2k px.
  try {
    const bmp = await createImageBitmap(file);
    const scale = IMAGE_MAX_SIDE / Math.max(bmp.width, bmp.height);
    if (scale >= 1) return file;
    const canvas = document.createElement("canvas");
    canvas.width = Math.round(bmp.width * scale);
    canvas.height = Math.round(bmp.height * scale);
    canvas.getContext("2d").drawImage(bmp, 0, 0, canvas.width, canvas.height);
    const blob = await new Promise((r) => canvas.toBlob(r, "image/jpeg", 0.88));
    return blob || file;
  } catch {
    return file;
  }
}

async function addFiles(files) {
  for (const file of files) {
    const isImage = file.type.startsWith("image/");
    const blob = isImage ? await shrinkImage(file) : file;
    attachments.push({
      name: file.name,
      type: isImage ? blob.type || file.type : file.type || "text/plain",
      data: await readAsBase64(blob),
    });
  }
  renderChips();
}

function submit() {
  const text = input.value.trim();
  if (!text && !attachments.length) return;
  if (!send({ type: "message", text, attachments })) return;
  input.value = "";
  attachments = [];
  renderChips();
  autosize();
}

// --- Session ----------------------------------------------------------------

function showLogin(info) {
  $("chat").hidden = true;
  $("login").hidden = false;
  const reason = new URLSearchParams(location.search).get("login");
  const notes = {
    invalid: "That sign-in couldn't be verified. Please try again.",
    unauthorized: "This Telegram account isn't allowed to use PaulusAI.",
  };
  $("login-note").textContent = notes[reason] || "";
  if (!info.bot) {
    $("login-note").textContent =
      "Sign-in isn't configured: the server couldn't determine the Telegram bot (set TELEGRAM_BOT_USERNAME).";
    return;
  }
  const box = $("tg-login");
  if (box.childElementCount) return;
  const s = document.createElement("script");
  s.async = true;
  s.src = "https://telegram.org/js/telegram-widget.js?22";
  s.dataset.telegramLogin = info.bot;
  s.dataset.size = "large";
  s.dataset.radius = "10";
  s.dataset.authUrl = new URL("auth", location.href).href;
  s.dataset.requestAccess = "write";
  box.append(s);
}

async function showChat(me) {
  $("login").hidden = true;
  $("chat").hidden = false;
  $("me").textContent = me.name;
  if (location.search) history.replaceState(null, "", location.pathname);
  try {
    const res = await fetch("api/history", { credentials: "same-origin" });
    if (res.ok) {
      const items = await res.json();
      for (const it of items) {
        if (it.role === "owner") ownerBubble(it.text, it.attachments || [], it.via);
        else agentBubble(it.text);
      }
      if (items.length) systemNote("— earlier conversation above —");
    }
  } catch (e) {
    console.error(e);
  }
  log.scrollTop = log.scrollHeight;
  connect();
  input.focus();
}

async function boot() {
  const res = await fetch("api/me", { credentials: "same-origin" });
  const info = await res.json();
  if (res.status === 401) showLogin(info);
  else showChat(info);
}

$("composer").addEventListener("submit", (e) => {
  e.preventDefault();
  submit();
});
input.addEventListener("input", autosize);
input.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey && !e.isComposing) {
    e.preventDefault();
    submit();
  }
});
fileInput.addEventListener("change", async () => {
  await addFiles([...fileInput.files]);
  fileInput.value = "";
});
input.addEventListener("paste", async (e) => {
  const files = [...(e.clipboardData?.files || [])];
  if (files.length) {
    e.preventDefault();
    await addFiles(files);
  }
});
$("logout").addEventListener("click", async () => {
  await fetch("api/logout", { method: "POST", credentials: "same-origin" });
  location.replace(location.pathname);
});

boot().catch((e) => {
  console.error(e);
  systemNote("Couldn't reach the server.");
});
