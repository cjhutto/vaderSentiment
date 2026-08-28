const test = require("node:test");
const assert = require("node:assert/strict");

const {
  fetchJsonWithRetry,
  mountResultHtml,
} = require("../static/analyzer/result-runtime.js");

test("mountResultHtml replaces an empty result container with a complete panel", () => {
  const container = {
    innerHTML: "",
    querySelector(selector) {
      return selector === "#verdict-box" && this.innerHTML.includes('id="verdict-box"')
        ? { id: "verdict-box" }
        : null;
    },
  };

  mountResultHtml(container, '<section id="verdict-box">Positive</section>');

  assert.equal(container.innerHTML, '<section id="verdict-box">Positive</section>');
});

test("mountResultHtml rejects an incomplete result fragment", () => {
  const container = {
    innerHTML: "",
    querySelector() {
      return null;
    },
  };

  assert.throws(
    () => mountResultHtml(container, "<p>Incomplete</p>"),
    /result view could not be loaded/i,
  );
});

test("fetchJsonWithRetry retries one transient connection failure", async () => {
  let calls = 0;
  const fetchImpl = async () => {
    calls += 1;
    if (calls === 1) throw new TypeError("Failed to fetch");
    return {
      ok: true,
      status: 200,
      async text() {
        return JSON.stringify({ status: "ok", result: { label: "positive" } });
      },
    };
  };

  const data = await fetchJsonWithRetry(fetchImpl, "/api/analyze/", {}, {
    retries: 1,
    retryDelayMs: 0,
  });

  assert.equal(calls, 2);
  assert.equal(data.result.label, "positive");
});

test("fetchJsonWithRetry reports an empty server response clearly", async () => {
  const fetchImpl = async () => ({
    ok: false,
    status: 502,
    async text() {
      return "";
    },
  });

  await assert.rejects(
    fetchJsonWithRetry(fetchImpl, "/api/analyze-url/", {}, { retries: 0 }),
    /empty response.*502/i,
  );
});
