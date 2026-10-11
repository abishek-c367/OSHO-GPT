const form = document.querySelector("#generate-form");
const message = document.querySelector("#message");
const button = document.querySelector("#generate-button");
const statusPill = document.querySelector("#status-pill");
const errorMessage = document.querySelector("#error-message");
const runMeta = document.querySelector("#run-meta");
const cards = [...document.querySelectorAll(".result-card")];

const controls = [
  ["temperature", "temperature-value", value => Number(value).toFixed(2)],
  ["max-new-tokens", "tokens-value", value => value],
  ["top-k", "top-k-value", value => value],
];
controls.forEach(([inputId, outputId, format]) => {
  const input = document.querySelector(`#${inputId}`);
  const output = document.querySelector(`#${outputId}`);
  const update = () => { output.value = format(input.value); output.textContent = format(input.value); };
  input.addEventListener("input", update);
  update();
});

function setLoading(loading) {
  button.disabled = loading;
  button.querySelector("span:first-child").textContent = loading ? "Generating..." : "Generate all three";
  statusPill.textContent = loading ? "Running" : "Ready";
  document.querySelectorAll(".result-card").forEach(card => card.classList.toggle("loading", loading));
  document.querySelectorAll(".model-status").forEach(status => { if (loading) status.textContent = "GENERATING"; });
}

function updateCard(event) {
  const card = cards.find(candidate => candidate.querySelector("h3").textContent === event.label);
  if (!card) return;
  const response = card.querySelector(".response");
  if (event.type === "start") {
    card.querySelector(".model-status").textContent = "GENERATING";
    response.textContent = "";
    response.classList.remove("empty-response");
    return;
  }
  response.textContent = event.text || "(No continuation returned.)";
  card.querySelector(".card-footer .metric:first-child").textContent = `${event.generated_tokens} tokens`;
  if (event.type === "done") {
    card.querySelector(".model-status").textContent = "COMPLETE";
    card.querySelector(".card-footer .metric:nth-child(2)").textContent = `${Math.round(event.duration_ms)} ms`;
    const copy = card.querySelector(".copy-button");
    copy.disabled = false;
    copy.onclick = async () => {
      await navigator.clipboard.writeText(response.textContent);
      copy.textContent = "Copied";
      setTimeout(() => { copy.textContent = "Copy"; }, 1200);
    };
  }
}

form.addEventListener("submit", async event => {
  event.preventDefault();
  errorMessage.hidden = true;
  const started = performance.now();
  setLoading(true);
  try {
    const response = await fetch("/generate/stream", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        message: message.value,
        temperature: Number(document.querySelector("#temperature").value),
        max_new_tokens: Number(document.querySelector("#max-new-tokens").value),
        top_k: Number(document.querySelector("#top-k").value),
        seed: document.querySelector("#use-seed").checked ? 1337 : null,
      }),
    });
    if (!response.ok || !response.body) throw new Error("Generation failed.");
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    while (true) {
      const { value, done } = await reader.read();
      buffer += decoder.decode(value || new Uint8Array(), { stream: !done });
      const events = buffer.split("\n\n");
      buffer = events.pop();
      for (const raw of events) {
        const line = raw.split("\n").find(item => item.startsWith("data: "));
        if (!line) continue;
        const streamEvent = JSON.parse(line.slice(6));
        if (streamEvent.type === "error") throw new Error(streamEvent.detail);
        updateCard(streamEvent);
      }
      if (done) break;
    }
    runMeta.textContent = `Completed in ${Math.round(performance.now() - started)} ms`;
  } catch (error) {
    errorMessage.textContent = error.message;
    errorMessage.hidden = false;
    statusPill.textContent = "Error";
    cards.forEach(card => { card.querySelector(".model-status").textContent = "ERROR"; });
  } finally {
    setLoading(false);
  }
});
