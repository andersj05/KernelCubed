const elements = {
  composer: document.querySelector("#composer"),
  input: document.querySelector("#message-input"),
  send: document.querySelector("#send-button"),
  messages: document.querySelector("#messages"),
  empty: document.querySelector("#empty-state"),
  notice: document.querySelector("#notice"),
  reset: document.querySelector("#reset-button"),
  statusDot: document.querySelector("#status-dot"),
  statusLabel: document.querySelector("#status-label"),
  speed: document.querySelector("#speed-value"),
  speedFill: document.querySelector("#speed-fill"),
  tokens: document.querySelector("#token-value"),
  time: document.querySelector("#time-value"),
  model: document.querySelector("#model-value"),
  backend: document.querySelector("#backend-value"),
  device: document.querySelector("#device-value"),
  custom: document.querySelector("#custom-value"),
  sdpa: document.querySelector("#sdpa-value"),
  context: document.querySelector("#context-label"),
};

let ready = false;
let generating = false;

function setNotice(message = "") {
  elements.notice.textContent = message;
  elements.notice.hidden = !message;
}

function updateSendState() {
  elements.send.disabled = !ready || generating || !elements.input.value.trim();
}

function resizeInput() {
  elements.input.style.height = "auto";
  elements.input.style.height = `${Math.min(elements.input.scrollHeight, 140)}px`;
}

function scrollToLatest() {
  elements.messages.scrollTo({
    top: elements.messages.scrollHeight,
    behavior: "smooth",
  });
}

function addMessage(role, text = "") {
  elements.empty?.remove();
  const article = document.createElement("article");
  article.className = `message ${role}`;
  if (role === "assistant") {
    const avatar = document.createElement("span");
    avatar.className = "message-avatar";
    avatar.setAttribute("aria-hidden", "true");
    avatar.textContent = "K?";
    article.appendChild(avatar);
  }
  const body = document.createElement("div");
  body.className = "message-body";
  const label = document.createElement("span");
  label.className = "message-role";
  label.textContent = role === "assistant" ? "Qwen" : "You";
  const content = document.createElement("div");
  content.className = "message-text";
  content.textContent = text;
  body.append(label, content);
  article.appendChild(body);
  elements.messages.appendChild(article);
  scrollToLatest();
  return { article, content };
}

function updateMetrics(event) {
  const speed = Number(event.tokens_per_second || 0);
  const tokens = Number(event.output_tokens || 0);
  const seconds = Number(event.elapsed_seconds || 0);
  elements.speed.textContent = speed ? speed.toFixed(1) : "?";
  elements.speedFill.style.width = `${Math.min(100, speed * 2.5)}%`;
  elements.tokens.textContent = tokens || "?";
  elements.time.textContent = seconds ? seconds.toFixed(2) : "?";
  if (event.type === "done") {
    elements.custom.textContent = event.custom_decode_calls.toLocaleString();
    elements.sdpa.textContent = event.sdpa_fallback_calls.toLocaleString();
  }
}

async function readEvents(response, onEvent) {
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  while (true) {
    const { value, done } = await reader.read();
    buffer += decoder.decode(value || new Uint8Array(), { stream: !done });
    const lines = buffer.split("\n");
    buffer = lines.pop() || "";
    for (const line of lines) {
      if (line) onEvent(JSON.parse(line));
    }
    if (done) break;
  }
  if (buffer.trim()) onEvent(JSON.parse(buffer));
}

async function sendMessage(message) {
  generating = true;
  setNotice();
  updateSendState();
  addMessage("user", message);
  const assistant = addMessage("assistant");
  assistant.article.classList.add("streaming");
  elements.input.value = "";
  resizeInput();

  try {
    const response = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message }),
    });
    if (!response.ok) {
      const payload = await response.json();
      throw new Error(payload.error || `Request failed (${response.status})`);
    }
    let text = "";
    let streamError = "";
    await readEvents(response, (event) => {
      if (event.type === "delta") {
        text += event.text;
        assistant.content.textContent = text;
        updateMetrics(event);
        scrollToLatest();
      } else if (event.type === "done") {
        assistant.content.textContent = event.text;
        updateMetrics(event);
      } else if (event.type === "error") {
        streamError = event.error;
      }
    });
    if (streamError) throw new Error(streamError);
  } catch (error) {
    assistant.article.remove();
    setNotice(error.message || "The response could not be generated.");
  } finally {
    assistant.article.classList.remove("streaming");
    generating = false;
    updateSendState();
    elements.input.focus();
  }
}

async function loadStatus() {
  try {
    const response = await fetch("/api/status", { cache: "no-store" });
    const status = await response.json();
    elements.backend.textContent = status.backend;
    elements.device.textContent = status.device;
    elements.context.textContent =
      `${status.max_context.toLocaleString()} context`;
    elements.model.textContent =
      status.model.split(/[\\/]/).filter(Boolean).pop() || status.model;
    elements.statusDot.className = `status-dot ${status.state}`;
    if (status.state === "ready") {
      ready = true;
      elements.statusLabel.textContent = "Model ready";
      setNotice();
      updateSendState();
      elements.input.focus();
      return;
    }
    if (status.state === "error") {
      elements.statusLabel.textContent = "Load failed";
      setNotice(status.error || "The model could not be loaded.");
      return;
    }
    elements.statusLabel.textContent = "Loading model";
    window.setTimeout(loadStatus, 750);
  } catch {
    elements.statusDot.className = "status-dot error";
    elements.statusLabel.textContent = "Disconnected";
    setNotice(
      "The local server is not responding. Restart it and refresh this page.",
    );
    window.setTimeout(loadStatus, 1500);
  }
}

elements.composer.addEventListener("submit", (event) => {
  event.preventDefault();
  const message = elements.input.value.trim();
  if (message && ready && !generating) sendMessage(message);
});

elements.input.addEventListener("input", () => {
  resizeInput();
  updateSendState();
});

elements.input.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    elements.composer.requestSubmit();
  }
});

elements.reset.addEventListener("click", async () => {
  if (generating) return;
  const response = await fetch("/api/reset", { method: "POST" });
  if (!response.ok) {
    const payload = await response.json();
    setNotice(payload.error || "The chat could not be reset.");
    return;
  }
  window.location.reload();
});

document.querySelectorAll(".suggestions button").forEach((button) => {
  button.addEventListener("click", () => {
    elements.input.value = button.textContent;
    resizeInput();
    updateSendState();
    elements.input.focus();
  });
});

loadStatus();
