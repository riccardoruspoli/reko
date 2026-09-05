import assert from "node:assert/strict";
import fs from "node:fs";
import vm from "node:vm";

class ClassList {
  constructor() {
    this.values = new Set(["hidden"]);
  }

  add(value) {
    this.values.add(value);
  }

  remove(value) {
    this.values.delete(value);
  }

  toggle(value, force) {
    if (force === undefined ? !this.values.has(value) : force) this.values.add(value);
    else this.values.delete(value);
  }

  contains(value) {
    return this.values.has(value);
  }
}

class Element {
  constructor(id) {
    this.id = id;
    this.value = "";
    this.checked = false;
    this.disabled = false;
    this.textContent = "";
    this.innerHTML = "";
    this.style = {};
    this.classList = new ClassList();
    this.listeners = new Map();
  }

  addEventListener(name, callback) {
    this.listeners.set(name, callback);
  }

  async emit(name, event = {}) {
    return this.listeners.get(name)?.(event);
  }

  async click() {
    return this.emit("click");
  }

  replaceChildren() {
    this.innerHTML = "";
  }

  append() {}

  appendChild() {}

  remove() {}
}

const ids = [
  "url", "summarize", "cancel-job", "retry-job", "status", "preview",
  "job-progress", "job-progress-message", "job-progress-count", "job-progress-bar",
  "provider", "model-name", "target-language", "temperature", "temperature-value",
  "max-retries", "length", "reasoning-effort", "thinking-toggle",
  "refresh-transcript-toggle", "include-summary", "include-key-points", "copy-markdown",
  "download-markdown", "stats-input-words", "stats-output-words", "stats-time-seconds",
];
const elements = new Map(ids.map((id) => [id, new Element(id)]));
const get = (id) => elements.get(id);
get("provider").value = "ollama";
get("model-name").value = "test";
get("target-language").value = "en";
get("temperature").value = "0";
get("max-retries").value = "0";
get("length").value = "medium";
get("include-summary").checked = true;
get("include-key-points").checked = true;

const fetches = [];
const intervals = [];
const timeouts = [];
const storage = new Map();
let nextResponse = null;

class FakeEventSource {
  static instances = [];

  constructor(url) {
    this.url = url;
    this.listeners = new Map();
    this.closed = false;
    FakeEventSource.instances.push(this);
  }

  addEventListener(name, callback) {
    this.listeners.set(name, callback);
  }

  close() {
    this.closed = true;
  }

  emit(name, data) {
    this.listeners.get(name)?.({ data: JSON.stringify(data) });
  }
}

const context = {
  Blob,
  EventSource: FakeEventSource,
  JSON,
  Math,
  Number,
  Object,
  String,
  URL: { createObjectURL: () => "blob:test", revokeObjectURL: () => {} },
  clearInterval: () => {},
  clearTimeout: () => {},
  document: {
    body: new Element("body"),
    getElementById: (id) => get(id),
    createElement: () => new Element("created"),
  },
  fetch: async (url, options = {}) => {
    fetches.push({ url, options });
    return nextResponse ?? { ok: true, json: async () => ({ ok: true, job: {} }) };
  },
  localStorage: {
    getItem: (key) => storage.get(key) ?? null,
    setItem: (key, value) => storage.set(key, value),
  },
  navigator: { clipboard: { writeText: async () => {} } },
  setInterval: (callback) => {
    intervals.push(callback);
    return intervals.length;
  },
  setTimeout: (callback) => {
    timeouts.push(callback);
    return timeouts.length;
  },
};
context.globalThis = context;
vm.runInNewContext(fs.readFileSync("src/reko/web/static/app.js", "utf8"), context);

await get("summarize").click();
assert.equal(get("status").textContent, "Missing URL");

get("refresh-transcript-toggle").checked = true;
await get("refresh-transcript-toggle").emit("change");
timeouts.at(-1)();
assert.equal(JSON.parse(storage.get("reko:ui:v1")).refreshTranscript, true);

get("url").value = "https://example.test/video";
nextResponse = {
  ok: true,
  json: async () => ({
    ok: true,
    job: { job_id: "job-1", state: "queued", phase: "queued", metrics: {} },
  }),
};
await get("summarize").click();
assert.equal(fetches.at(-1).url, "/api/jobs");
assert.equal(fetches.at(-1).options.method, "POST");
assert.equal(get("summarize").disabled, true);
assert.equal(FakeEventSource.instances.at(-1).url, "/api/jobs/job-1/events");

const source = FakeEventSource.instances.at(-1);
source.emit("progress", {
  state: "running", phase: "summarizing", message: "Working", completed: 1, total: 2,
  metrics: { input_words: 100, output_words: 20 }, elapsed_seconds: 1,
});
assert.equal(get("job-progress-count").textContent, "1/2");
assert.equal(get("job-progress-bar").style.width, "50%");

source.emit("terminal", {
  state: "succeeded", phase: "completed", message: "Done", metrics: {}, elapsed_seconds: 1,
  result: { markdown: "# Result", html: "<h1>Result</h1>", video_id: "video-1" },
});
assert.equal(get("preview").innerHTML, "<h1>Result</h1>");
assert.equal(get("summarize").disabled, false);
assert.equal(source.closed, true);

get("url").value = "https://example.test/second";
nextResponse = {
  ok: true,
  json: async () => ({ ok: true, job: { job_id: "job-2", state: "queued", phase: "queued", metrics: {} } }),
};
await get("summarize").click();
const failingSource = FakeEventSource.instances.at(-1);
failingSource.onerror();
assert.equal(intervals.length, 1);
nextResponse = {
  ok: true,
  json: async () => ({
    ok: true,
    job: { state: "cancelled", phase: "cancelled", message: "Cancelled", metrics: {}, elapsed_seconds: 1 },
  }),
};
await intervals[0]();
assert.equal(get("status").textContent, "Cancelled");
assert.equal(get("retry-job").classList.contains("hidden"), false);

console.log("web UI scenarios passed");
