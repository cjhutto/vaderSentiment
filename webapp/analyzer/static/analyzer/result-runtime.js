(function exposeResultRuntime(globalScope) {
  function mountResultHtml(container, resultHtml) {
    if (!container || typeof resultHtml !== "string" || !resultHtml.trim()) {
      throw new Error("Analysis succeeded, but the result view could not be loaded.");
    }

    container.innerHTML = resultHtml;
    if (!container.querySelector("#verdict-box")) {
      container.innerHTML = "";
      throw new Error("Analysis succeeded, but the result view could not be loaded.");
    }

    return container;
  }

  function wait(milliseconds) {
    return new Promise(resolve => setTimeout(resolve, milliseconds));
  }

  async function fetchJsonWithRetry(fetchImpl, url, options = {}, config = {}) {
    const retries = Number.isInteger(config.retries) ? config.retries : 1;
    const retryDelayMs = Number.isFinite(config.retryDelayMs) ? config.retryDelayMs : 350;
    let response;

    for (let attempt = 0; attempt <= retries; attempt += 1) {
      try {
        response = await fetchImpl(url, options);
        break;
      } catch (error) {
        if (attempt >= retries) {
          throw new Error(
            "Unable to reach the analysis server. It may be restarting; wait a moment and try again.",
          );
        }
        await wait(retryDelayMs);
      }
    }

    const rawBody = await response.text();
    if (!rawBody.trim()) {
      throw new Error(`Analysis server returned an empty response (HTTP ${response.status}).`);
    }

    let data;
    try {
      data = JSON.parse(rawBody);
    } catch (error) {
      throw new Error(`Analysis server returned an invalid response (HTTP ${response.status}).`);
    }

    if (!response.ok || data.status !== "ok") {
      throw new Error(data.message || `Analysis failed (HTTP ${response.status}).`);
    }

    return data;
  }

  const runtime = { fetchJsonWithRetry, mountResultHtml };
  globalScope.VaderResultRuntime = runtime;

  if (typeof module !== "undefined" && module.exports) {
    module.exports = runtime;
  }
})(typeof window !== "undefined" ? window : globalThis);
