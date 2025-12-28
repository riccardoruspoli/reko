const urlInput = document.getElementById("url");
const summarizeButton = document.getElementById("summarize");
const statusBadge = document.getElementById("status");
const previewEl = document.getElementById("preview");

const providerInput = document.getElementById("provider");
const modelNameInput = document.getElementById("model-name");
const targetLanguageInput = document.getElementById("target-language");
const temperatureInput = document.getElementById("temperature");
const temperatureValue = document.getElementById("temperature-value");
const targetChunkWordsInput = document.getElementById("target-chunk-words");
const maxTokensInput = document.getElementById("max-tokens");
const maxRetriesInput = document.getElementById("max-retries");
const lengthSelect = document.getElementById("length");
const thinkToggle = document.getElementById("thinking-toggle");
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
    if (timer) window.clearTimeout(timer);
    timer = window.setTimeout(() => fn(...args), delayMs);
  };
}

function loadCachedSettings() {
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY);
    if (!raw) return null;
    const parsed = JSON.parse(raw);
    return parsed && typeof parsed === "object" ? parsed : null;
  } catch {
    return null;
  }
}

function saveCachedSettings(settings) {
  try {
    window.localStorage.setItem(STORAGE_KEY, JSON.stringify(settings));
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
    temperature: readNumber(temperatureInput, 1),
    targetChunkWords: readNumber(targetChunkWordsInput, 800),
    maxTokens: readNumber(maxTokensInput, 16384),
    maxRetries: readNumber(maxRetriesInput, 3),
    think: Boolean(thinkToggle?.checked),
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
  if (temperatureInput && settings.temperature != null) {
    temperatureInput.value = String(settings.temperature);
  }
  if (targetChunkWordsInput && settings.targetChunkWords != null) {
    targetChunkWordsInput.value = String(settings.targetChunkWords);
  }
  if (maxTokensInput && settings.maxTokens != null) {
    maxTokensInput.value = String(settings.maxTokens);
  }
  if (maxRetriesInput && settings.maxRetries != null) {
    maxRetriesInput.value = String(settings.maxRetries);
  }
  if (thinkToggle && typeof settings.think === "boolean") {
    thinkToggle.checked = settings.think;
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
  window.setTimeout(() => {
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
  if (navigator.clipboard?.writeText) {
    try {
      await navigator.clipboard.writeText(text);
      return true;
    } catch {
      // fall back below
    }
  }

  const textarea = document.createElement("textarea");
  textarea.value = text;
  textarea.setAttribute("readonly", "");
  textarea.style.position = "fixed";
  textarea.style.left = "-9999px";
  textarea.style.top = "-9999px";
  document.body.appendChild(textarea);
  textarea.select();

  try {
    return document.execCommand("copy");
  } finally {
    textarea.remove();
  }
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
    targetChunkWords: readNumber(targetChunkWordsInput, 800),
    maxTokens: readNumber(maxTokensInput, 16384),
    maxRetries: readNumber(maxRetriesInput, 3),
    think: Boolean(thinkToggle?.checked),
    includeSummary: Boolean(includeSummaryToggle?.checked ?? true),
    includeKeyPoints: Boolean(includeKeyPointsToggle?.checked ?? true),
    length: lengthSelect?.value || "medium",
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
        data.error || data.detail || `Request failed (${resp.status})`
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
  temperatureInput,
  targetChunkWordsInput,
  maxTokensInput,
  maxRetriesInput,
  thinkToggle,
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
    } catch {
      // ignore
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
