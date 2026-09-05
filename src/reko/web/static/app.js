const urlInput = document.getElementById("url");
const summarizeButton = document.getElementById("summarize");
const cancelJobButton = document.getElementById("cancel-job");
const retryJobButton = document.getElementById("retry-job");
const statusBadge = document.getElementById("status");
const previewEl = document.getElementById("preview");
const progressEl = document.getElementById("job-progress");
const progressMessageEl = document.getElementById("job-progress-message");
const progressCountEl = document.getElementById("job-progress-count");
const progressBarEl = document.getElementById("job-progress-bar");

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
let activeJobId = null;
let eventSource = null;
let pollingTimer = null;
let sseRetryTimer = null;

const STORAGE_KEY = "reko:ui:v1";

function setStatus(text) {
  if (!statusBadge) return;
  statusBadge.textContent = text;
}

function showPreviewError(message) {
  previewEl.replaceChildren();
  const error = document.createElement("pre");
  error.textContent = message;
  previewEl.append(error);
}

function setJobControls(active) {
  summarizeButton.disabled = active;
  if (cancelJobButton) {
    cancelJobButton.disabled = !active;
    cancelJobButton.classList.toggle("hidden", !active);
    cancelJobButton.classList.toggle("flex", active);
  }
  if (retryJobButton) {
    retryJobButton.classList.toggle("hidden", active);
    retryJobButton.classList.remove("flex");
  }
}

function setRetryVisible(visible) {
  if (!retryJobButton) return;
  retryJobButton.classList.toggle("hidden", !visible);
  retryJobButton.classList.toggle("flex", visible);
}

function resetProgress() {
  if (progressEl) progressEl.classList.add("hidden");
  if (progressBarEl) progressBarEl.style.width = "0%";
  if (progressCountEl) progressCountEl.textContent = "";
}

function updateProgress(job) {
  if (!job || !progressEl) return;
  progressEl.classList.remove("hidden");
  if (progressMessageEl) progressMessageEl.textContent = job.message || "Working";

  const completed = Number(job.completed);
  const total = Number(job.total);
  const hasProgress = Number.isFinite(completed) && Number.isFinite(total) && total > 0;
  if (progressCountEl) {
    progressCountEl.textContent = hasProgress ? `${completed}/${total}` : job.phase || "";
  }
  if (progressBarEl) {
    const percent = hasProgress ? Math.min(100, Math.max(0, (completed / total) * 100)) : 0;
    progressBarEl.style.width = `${percent}%`;
  }

  setStatus(job.state === "succeeded" ? "Done" : job.state || "Running");
  setStats({
    inputWords: job.metrics?.input_words,
    outputWords: job.metrics?.output_words,
    elapsedSeconds: job.metrics?.elapsed_seconds ?? job.elapsed_seconds,
  });
}

function stopJobUpdates() {
  if (eventSource) {
    eventSource.close();
    eventSource = null;
  }
  if (pollingTimer) {
    globalThis.clearInterval(pollingTimer);
    pollingTimer = null;
  }
  if (sseRetryTimer) {
    globalThis.clearTimeout(sseRetryTimer);
    sseRetryTimer = null;
  }
}

function finishJob(job) {
  stopJobUpdates();
  activeJobId = null;
  setJobControls(false);
  setRetryVisible(["failed", "cancelled"].includes(job.state));
  updateProgress(job);

  if (job.state === "succeeded") {
    previewEl.innerHTML = job.result?.html || "";
    lastMarkdown = job.result?.markdown || "";
    lastVideoId = job.result?.video_id || "";
    return;
  }
  if (job.state === "cancelled") {
    setStatus("Cancelled");
    showPreviewError("The job was cancelled.");
    return;
  }
  if (job.state === "failed") {
    setStatus("Error");
    showPreviewError(job.error || "The job failed.");
  }
}

function applyJobSnapshot(job) {
  updateProgress(job);
  if (["succeeded", "failed", "cancelled"].includes(job.state)) {
    finishJob(job);
  }
}

async function pollJob() {
  if (!activeJobId) return;
  try {
    const response = await fetch(`/api/jobs/${activeJobId}`);
    const data = await response.json();
    if (!response.ok || data.ok === false) throw new Error(data.error || "Job lookup failed");
    applyJobSnapshot(data.job);
  } catch (error) {
    stopJobUpdates();
    activeJobId = null;
    setJobControls(false);
    setStatus("Error");
    showPreviewError(error?.message || String(error));
  }
}

function startPollingFallback() {
  if (pollingTimer || !activeJobId) return;
  pollingTimer = globalThis.setInterval(pollJob, 1000);
  pollJob();
}

function retrySseConnection() {
  if (sseRetryTimer || !activeJobId || eventSource) return;
  sseRetryTimer = globalThis.setTimeout(() => {
    sseRetryTimer = null;
    if (activeJobId && !eventSource) startJobUpdates(activeJobId);
  }, 5000);
}

function startJobUpdates(jobId) {
  if (!globalThis.EventSource) {
    startPollingFallback();
    return;
  }
  eventSource = new EventSource(`/api/jobs/${jobId}/events`);
  eventSource.onopen = () => {
    if (pollingTimer) {
      globalThis.clearInterval(pollingTimer);
      pollingTimer = null;
    }
  };
  ["state", "progress", "terminal"].forEach((eventName) => {
    eventSource.addEventListener(eventName, (event) => {
      try {
        applyJobSnapshot(JSON.parse(event.data));
      } catch {
        startPollingFallback();
      }
    });
  });
  eventSource.onerror = () => {
    if (eventSource) {
      eventSource.close();
      eventSource = null;
    }
    if (activeJobId) {
      startPollingFallback();
      retrySseConnection();
    }
  };
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
  setJobControls(true);
  setRetryVisible(false);
  setStatus("Queued");
  previewEl.innerHTML = "<p>Working...</p>";
  resetProgress();
  lastMarkdown = "";
  lastVideoId = "";
  setStats({ inputWords: null, outputWords: null, elapsedSeconds: null });

  try {
    const resp = await fetch("/api/jobs", {
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

    activeJobId = data.job?.job_id;
    if (!activeJobId) throw new Error("The server did not return a job ID.");
    applyJobSnapshot(data.job);
    if (activeJobId) startJobUpdates(activeJobId);
  } catch (err) {
    setStatus("Error");
    showPreviewError(err?.message || String(err));
    activeJobId = null;
    setJobControls(false);
  }
}

summarizeButton.addEventListener("click", summarize);
urlInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter") summarize();
});

if (cancelJobButton) {
  cancelJobButton.addEventListener("click", async () => {
    if (!activeJobId) return;
    cancelJobButton.disabled = true;
    setStatus("Cancelling");
    try {
      const response = await fetch(`/api/jobs/${activeJobId}`, { method: "DELETE" });
      const data = await response.json();
      if (!response.ok || data.ok === false) throw new Error(data.error || "Cancellation failed");
      applyJobSnapshot(data.job);
    } catch (error) {
      cancelJobButton.disabled = false;
      setStatus("Error");
      showPreviewError(error?.message || String(error));
    }
  });
}

if (retryJobButton) {
  retryJobButton.addEventListener("click", summarize);
}

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
