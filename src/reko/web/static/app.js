const urlInput = document.getElementById("url");
const summarizeButton = document.getElementById("summarize");
const statusBadge = document.getElementById("status");
const previewEl = document.getElementById("preview");

const providerInput = document.getElementById("provider");
const modelNameInput = document.getElementById("model-name");
const targetLanguageInput = document.getElementById("target-language");
const temperatureInput = document.getElementById("temperature");
const temperatureValue = document.getElementById("temperature-value");
const maxRetriesInput = document.getElementById("max-retries");
const lengthSelect = document.getElementById("length");
const reasoningEffortSelect = document.getElementById("reasoning-effort");
const thinkToggle = document.getElementById("thinking-toggle");
const refreshTranscriptToggle = document.getElementById(
  "refresh-transcript-toggle",
);
const includeSummaryToggle = document.getElementById("include-summary");
const includeKeyPointsToggle = document.getElementById("include-key-points");
const copyMarkdownButton = document.getElementById("copy-markdown");
const downloadMarkdownButton = document.getElementById("download-markdown");
const inputWordsEl = document.getElementById("stats-input-words");
const outputWordsEl = document.getElementById("stats-output-words");
const timeSecondsEl = document.getElementById("stats-time-seconds");

let lastMarkdown = "";
let lastVideoId = "";

const STORAGE_KEY = "reko:ui:v1";

function setStatus(text) {
  if (!statusBadge) return;
  statusBadge.textContent = text;
}

function debounce(fn, delayMs) {
  let timer = null;
  return (...args) => {
    if (timer) globalThis.clearTimeout(timer);
    timer = globalThis.setTimeout(() => fn(...args), delayMs);
  };
}

function loadCachedSettings() {
  try {
    const raw = globalThis.localStorage.getItem(STORAGE_KEY);
    if (!raw) return null;
    const parsed = JSON.parse(raw);
    return parsed && typeof parsed === "object" ? parsed : null;
  } catch {
    return null;
  }
}

function saveCachedSettings(settings) {
  try {
    globalThis.localStorage.setItem(STORAGE_KEY, JSON.stringify(settings));
  } catch {
    // ignore storage errors (private mode, quota, etc.)
  }
}

function readSettingsFromForm() {
  return {
    url: (urlInput?.value || "").trim(),
    provider: (providerInput?.value || "").trim(),
    modelName: (modelNameInput?.value || "").trim(),
    targetLanguage: (targetLanguageInput?.value || "").trim(),
    length: lengthSelect?.value || "medium",
    reasoningEffort: reasoningEffortSelect?.value || "",
    temperature: readNumber(temperatureInput, 1),
    maxRetries: readNumber(maxRetriesInput, 3),
    think: Boolean(thinkToggle?.checked),
    refreshTranscript: Boolean(refreshTranscriptToggle?.checked),
    includeSummary: Boolean(includeSummaryToggle?.checked ?? true),
    includeKeyPoints: Boolean(includeKeyPointsToggle?.checked ?? true),
  };
}

function applySettingsToForm(settings) {
  if (!settings || typeof settings !== "object") return;

  if (urlInput && typeof settings.url === "string")
    urlInput.value = settings.url;
  if (providerInput && typeof settings.provider === "string") {
    providerInput.value = settings.provider;
  }
  if (modelNameInput && typeof settings.modelName === "string") {
    modelNameInput.value = settings.modelName;
  }
  if (targetLanguageInput && typeof settings.targetLanguage === "string") {
    targetLanguageInput.value = settings.targetLanguage;
  }
  if (lengthSelect && typeof settings.length === "string") {
    lengthSelect.value = settings.length;
  }
  if (reasoningEffortSelect && typeof settings.reasoningEffort === "string") {
    reasoningEffortSelect.value = settings.reasoningEffort;
  }
  if (temperatureInput && settings.temperature != null) {
    temperatureInput.value = String(settings.temperature);
  }
  if (maxRetriesInput && settings.maxRetries != null) {
    maxRetriesInput.value = String(settings.maxRetries);
  }
  if (thinkToggle && typeof settings.think === "boolean") {
    thinkToggle.checked = settings.think;
  }
  if (
    refreshTranscriptToggle &&
    typeof settings.refreshTranscript === "boolean"
  ) {
    refreshTranscriptToggle.checked = settings.refreshTranscript;
  }
  if (includeSummaryToggle && typeof settings.includeSummary === "boolean") {
    includeSummaryToggle.checked = settings.includeSummary;
  }
  if (
    includeKeyPointsToggle &&
    typeof settings.includeKeyPoints === "boolean"
  ) {
    includeKeyPointsToggle.checked = settings.includeKeyPoints;
  }

  if (temperatureInput && temperatureValue) {
    temperatureValue.textContent = temperatureInput.value;
  }
}

function flashStatus(text, ms = 1200) {
  if (!statusBadge) return;
  const previous = statusBadge.textContent;
  statusBadge.textContent = text;
  globalThis.setTimeout(() => {
    statusBadge.textContent = previous;
  }, ms);
}

function setStats({ inputWords, outputWords, elapsedSeconds }) {
  if (inputWordsEl) {
    inputWordsEl.textContent =
      typeof inputWords === "number" ? inputWords.toLocaleString() : "—";
  }
  if (outputWordsEl) {
    outputWordsEl.textContent =
      typeof outputWords === "number" ? outputWords.toLocaleString() : "—";
  }
  if (timeSecondsEl) {
    timeSecondsEl.textContent =
      typeof elapsedSeconds === "number" ? elapsedSeconds.toFixed(1) : "—";
  }
}

async function copyToClipboard(text) {
  if (!navigator.clipboard?.writeText) {
    return false;
  }

  await navigator.clipboard.writeText(text);
  return true;
}

function readNumber(inputEl, fallback) {
  const parsed = Number((inputEl?.value || "").toString().trim());
  return Number.isFinite(parsed) ? parsed : fallback;
}

function buildConfigPayload() {
  return {
    provider: (providerInput?.value || "").trim(),
    modelName: (modelNameInput?.value || "").trim(),
    targetLanguage: (targetLanguageInput?.value || "").trim(),
    temperature: readNumber(temperatureInput, 1),
    maxRetries: readNumber(maxRetriesInput, 3),
    think: Boolean(thinkToggle?.checked),
    refreshTranscript: Boolean(refreshTranscriptToggle?.checked),
    includeSummary: Boolean(includeSummaryToggle?.checked ?? true),
    includeKeyPoints: Boolean(includeKeyPointsToggle?.checked ?? true),
    length: lengthSelect?.value || "medium",
    reasoningEffort: reasoningEffortSelect?.value || null,
  };
}

async function summarize() {
  const url = (urlInput.value || "").trim();
  if (!url) {
    setStatus("Missing URL");
    return;
  }

  saveCachedSettings(readSettingsFromForm());
  summarizeButton.disabled = true;
  setStatus("Running");
  previewEl.innerHTML = "<p>Working...</p>";
  lastMarkdown = "";
  lastVideoId = "";
  setStats({ inputWords: null, outputWords: null, elapsedSeconds: null });

  try {
    const resp = await fetch("/api/summarize", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ url, config: buildConfigPayload() }),
    });

    const data = await resp.json().catch(() => ({}));
    if (!resp.ok || data.ok === false) {
      throw new Error(
        data.error || data.detail || `Request failed (${resp.status})`,
      );
    }

    previewEl.innerHTML = data.html || "";
    lastMarkdown = data.markdown || "";
    lastVideoId = data.video_id || "";
    setStatus("Done");
    setStats({
      inputWords: data.stats?.input_words,
      outputWords: data.stats?.output_words,
      elapsedSeconds: data.stats?.elapsed_seconds,
    });
  } catch (err) {
    setStatus("Error");
    previewEl.innerHTML = `<pre>${err?.message || String(err)}</pre>`;
  } finally {
    summarizeButton.disabled = false;
  }
}

summarizeButton.addEventListener("click", summarize);
urlInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter") summarize();
});

const persistSettingsDebounced = debounce(() => {
  saveCachedSettings(readSettingsFromForm());
}, 300);

function bindPersist(el) {
  if (!el) return;
  el.addEventListener("input", persistSettingsDebounced);
  el.addEventListener("change", persistSettingsDebounced);
}

applySettingsToForm(loadCachedSettings());

[
  urlInput,
  providerInput,
  modelNameInput,
  targetLanguageInput,
  lengthSelect,
  reasoningEffortSelect,
  temperatureInput,
  maxRetriesInput,
  thinkToggle,
  refreshTranscriptToggle,
  includeSummaryToggle,
  includeKeyPointsToggle,
].forEach(bindPersist);

if (temperatureInput && temperatureValue) {
  temperatureValue.textContent = temperatureInput.value;
  temperatureInput.addEventListener("input", () => {
    temperatureValue.textContent = temperatureInput.value;
  });
}

if (copyMarkdownButton) {
  copyMarkdownButton.addEventListener("click", async () => {
    if (!lastMarkdown) return;
    try {
      const ok = await copyToClipboard(lastMarkdown);
      if (ok) flashStatus("Copied");
      else flashStatus("Copy failed");
    } catch {
      flashStatus("Copy failed");
    }
  });
}

if (downloadMarkdownButton) {
  downloadMarkdownButton.addEventListener("click", () => {
    if (!lastMarkdown) return;
    const blob = new Blob([lastMarkdown], {
      type: "text/markdown;charset=utf-8",
    });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = `${lastVideoId || "reko-summary"}.md`;
    document.body.appendChild(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(a.href), 0);
  });
}
