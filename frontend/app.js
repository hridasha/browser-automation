// Override by setting these before this script loads, e.g.:
// <script>
//   window.BROWSER_AGENT_API = "https://your-api.example.com";
//   window.BROWSER_AGENT_API_KEY = "...";  // only needed if the server sets API_KEY
// </script>
const API = window.BROWSER_AGENT_API || "http://localhost:8000";
const API_KEY = window.BROWSER_AGENT_API_KEY || null;

function apiHeaders(extra = {}) {
  return API_KEY ? { ...extra, "X-API-Key": API_KEY } : extra;
}

let lastResults = [];
let pendingClarifyGoal = null;

// ── Theme ──
function applyTheme(theme) {
  document.documentElement.dataset.theme = theme;
  const btn = document.getElementById("themeToggle");
  if (btn) btn.textContent = theme === "light" ? "Dark" : "Light";
}

function toggleTheme() {
  const next = document.documentElement.dataset.theme === "light" ? "dark" : "light";
  applyTheme(next);
  try { localStorage.setItem("theme", next); } catch {}
}

(function initTheme() {
  let saved = null;
  try { saved = localStorage.getItem("theme"); } catch {}
  applyTheme(saved || "dark");
})();

// ── Server health check ──
async function checkServer() {
  try {
    const res = await fetch(`${API}/`);
    setLinkStatus(res.ok ? "online" : "error", res.ok ? "Live" : "Error");
  } catch {
    setLinkStatus("offline", "Offline");
  }
}

function setLinkStatus(state, label) {
  document.getElementById("serverStatus").dataset.state = state;
  document.getElementById("linkLabel").textContent = label;
}

function escapeHtml(str) {
  const div = document.createElement("div");
  div.textContent = str ?? "";
  return div.innerHTML;
}

// Only allow http(s) URLs in href attributes — scraped/LLM content is untrusted and
// could otherwise smuggle a javascript: URI into a rendered link.
function safeUrl(value) {
  try {
    const u = new URL(String(value));
    return (u.protocol === "http:" || u.protocol === "https:") ? u.href : "#";
  } catch {
    return "#";
  }
}

// ── Logging ──
function log(msg, type = "info") {
  const box = document.getElementById("logBox");
  const entry = document.createElement("div");
  entry.className = `log-entry ${type}`;
  const time = new Date().toLocaleTimeString([], { hour12: false });
  entry.innerHTML = `<span class="ts">${time}</span>${escapeHtml(msg)}`;
  box.appendChild(entry);
  box.scrollTop = box.scrollHeight;
}

function clearLogs() {
  document.getElementById("logBox").innerHTML = "";
}

// ============================================================
// PIPELINE RAIL — reflects the real planner/search/browser/
// extraction/validator nodes as SSE 'node' events arrive.
// ============================================================
const RAIL_ORDER = ["planner", "search", "browser", "extraction", "validator"];

function railNode(key) {
  return document.querySelector(`.rail-node[data-node="${key}"]`);
}

function setRailMeta(key, text) {
  const el = document.getElementById(`railMeta-${key}`);
  if (el) el.textContent = text;
}

function resetRail() {
  document.querySelectorAll(".rail-node").forEach(n => n.classList.remove("active", "done", "paused", "error"));
  document.querySelectorAll(".rail-connector").forEach(c => c.classList.remove("done", "flowing"));
  RAIL_ORDER.forEach(key => setRailMeta(key, ""));
  document.getElementById("railStatus").textContent = "idle — waiting for a goal";
  document.getElementById("attemptsList").innerHTML = "";
}

// ── Retry-attempt timeline ──
function appendAttempt(attempt, query) {
  const list = document.getElementById("attemptsList");
  if (list.querySelector(`[data-attempt="${attempt}"]`)) return; // already recorded
  const row = document.createElement("div");
  row.className = "attempt-item";
  row.dataset.attempt = attempt;
  row.innerHTML = `<span class="attempt-num">#${attempt}</span>${escapeHtml(query)}`;
  list.appendChild(row);
}

function setRailState(activeKey) {
  const idx = RAIL_ORDER.indexOf(activeKey);
  RAIL_ORDER.forEach((key, i) => {
    const node = railNode(key);
    node.classList.remove("active", "done", "paused", "error");
    if (i < idx) node.classList.add("done");
    else if (i === idx) node.classList.add("active");
  });
  document.querySelectorAll(".rail-connector").forEach((c, i) => {
    c.classList.remove("done", "flowing");
    if (i < idx - 1) c.classList.add("done");
    else if (i === idx - 1) c.classList.add("flowing");
  });
}

function setRailAllDone() {
  RAIL_ORDER.forEach(key => {
    const node = railNode(key);
    node.classList.remove("active", "paused", "error");
    node.classList.add("done");
  });
  document.querySelectorAll(".rail-connector").forEach(c => {
    c.classList.remove("flowing");
    c.classList.add("done");
  });
}

function setRailPausedAt(key) {
  setRailState(key);
  railNode(key).classList.remove("active");
  railNode(key).classList.add("paused");
  document.querySelectorAll(".rail-connector").forEach(c => c.classList.remove("flowing"));
}

function setRailErrorAt(key) {
  setRailState(key);
  railNode(key).classList.remove("active");
  railNode(key).classList.add("error");
}

// ── Tool tag ──
function showToolTag(tool) {
  const tag = document.getElementById("toolTag");
  tag.textContent = tool;
  tag.hidden = false;
}

// ── Main: streaming auto function ──
async function runAuto() {
  const input = document.getElementById("goalInput").value.trim();
  if (!input) { document.getElementById("goalInput").focus(); return; }

  const btn = document.getElementById("runBtn");
  btn.disabled = true;
  btn.textContent = "Running…";

  clearLogs();
  resetRail();
  document.getElementById("resultsBox").innerHTML = '<p class="empty">Agent is running…</p>';
  document.getElementById("toolOutputCard").hidden = true;
  document.getElementById("toolTag").hidden = true;
  document.getElementById("exportButtons").hidden = true;
  document.getElementById("clarifyPanel").hidden = true;
  lastResults = [];
  pendingClarifyGoal = null;

  log(`Goal: ${input}`, "info");

  try {
    const response = await fetch(`${API}/auto/stream`, {
      method: "POST",
      headers: apiHeaders({ "Content-Type": "application/json" }),
      body: JSON.stringify({ input }),
    });

    if (!response.ok) {
      throw new Error(`Server error: ${response.status}`);
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });

      // SSE lines come as "data: {...}\n\n"
      const lines = buffer.split("\n");
      buffer = lines.pop(); // keep incomplete last line

      for (const line of lines) {
        if (!line.startsWith("data: ")) continue;
        const raw = line.slice(6).trim();
        if (!raw) continue;

        let evt;
        try { evt = JSON.parse(raw); } catch { continue; }

        handleEvent(evt);
      }
    }

  } catch (err) {
    log(`Error: ${err.message}`, "error");
    document.getElementById("railStatus").textContent = "error";
  } finally {
    btn.disabled = false;
    btn.textContent = "Run →";
  }
}

// ── Handle each SSE event ──
function handleEvent(evt) {
  if (evt.event === "route") {
    showToolTag(evt.tool);
    log(`Routed → ${evt.tool}`, "ok");
    if (evt.tool !== "run_agent") {
      document.getElementById("railStatus").textContent = `${evt.tool} — pipeline not used for this request`;
    }
  }

  else if (evt.event === "node") {
    setRailState(evt.node);
    document.getElementById("railStatus").textContent = evt.status || evt.node;
    if (evt.node === "browser") setRailMeta("browser", `${evt.pages} page${evt.pages === 1 ? "" : "s"}`);
    if (evt.node === "extraction") setRailMeta("extraction", `${evt.items} record${evt.items === 1 ? "" : "s"}`);
    if (evt.node === "validator") setRailMeta("validator", `${evt.items} kept`);
    if (evt.node === "search" && evt.attempt) appendAttempt(evt.attempt, evt.search_query || "");
    log(evt.label, "info");
  }

  else if (evt.event === "needs_input") {
    pendingClarifyGoal = evt.goal;
    document.getElementById("clarifyQuestion").textContent = evt.question;
    document.getElementById("clarifyPanel").hidden = false;
    document.getElementById("clarifyInput").focus();
    setRailPausedAt("planner");
    document.getElementById("railStatus").textContent = "awaiting your answer";
    log(evt.question, "warn");
  }

  else if (evt.event === "done") {
    const tool = evt.tool_used;

    if (tool === "run_agent") {
      if (evt.status === "error") {
        setRailErrorAt("validator");
        document.getElementById("railStatus").textContent = "error — gave up after 3 attempts";
      } else {
        setRailAllDone();
        document.getElementById("railStatus").textContent = "done";
      }
      renderResults(evt.jobs, evt.goal);
      setExportable(evt.jobs);
      log(`Done — ${evt.jobs_found} result(s) across ${evt.pages_visited} page(s)`, "ok");
      loadRunHistory();

    } else if (tool === "search_web") {
      renderSearchResults(evt.results);
      setExportable(evt.results);
      log(`Found ${evt.total} search results`, "ok");

    } else if (tool === "browse_page") {
      showRawOutput(`URL: ${evt.url}\nLength: ${evt.content_length} chars\n\n${evt.content}`);
      log(`Fetched ${evt.content_length} chars from page`, "ok");
    }
  }

  else if (evt.event === "error") {
    log(evt.message, "error");
    document.getElementById("railStatus").textContent = "error";
  }
}

// ── Human-in-the-loop: answer a clarifying question ──
async function answerClarify() {
  const input = document.getElementById("clarifyInput");
  const answer = input.value.trim();
  if (!answer || !pendingClarifyGoal) return;

  const btn = document.querySelector("#clarifyPanel .btn-run");
  btn.disabled = true;
  log(`Answer: ${answer}`, "info");

  try {
    const res = await fetch(`${API}/run/clarify`, {
      method: "POST",
      headers: apiHeaders({ "Content-Type": "application/json" }),
      body: JSON.stringify({ goal: pendingClarifyGoal, answer }),
    });
    if (!res.ok) throw new Error(`Server error: ${res.status}`);
    const data = await res.json();

    input.value = "";

    if (data.status === "needs_input") {
      document.getElementById("clarifyQuestion").textContent = data.human_question;
      log(data.human_question, "warn");
      return;
    }

    document.getElementById("clarifyPanel").hidden = true;
    pendingClarifyGoal = null;

    if (data.status === "error") {
      setRailErrorAt("validator");
      document.getElementById("railStatus").textContent = "error — gave up after 3 attempts";
    } else {
      setRailAllDone();
      setRailMeta("browser", `${data.pages_visited} page${data.pages_visited === 1 ? "" : "s"}`);
      setRailMeta("validator", `${data.jobs_found} kept`);
      document.getElementById("railStatus").textContent = "done";
    }

    renderResults(data.jobs, data.goal);
    setExportable(data.jobs);
    log(`Done — ${data.jobs_found} result(s) across ${data.pages_visited} page(s)`, "ok");
    loadRunHistory();

  } catch (err) {
    log(`Error: ${err.message}`, "error");
  } finally {
    btn.disabled = false;
  }
}

// ── Export ──
function setExportable(items) {
  lastResults = items || [];
  document.getElementById("exportButtons").hidden = lastResults.length === 0;
}

function downloadFile(filename, content, mime) {
  const blob = new Blob([content], { type: mime });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}

function exportResults(format) {
  if (!lastResults.length) return;

  if (format === "json") {
    downloadFile("results.json", JSON.stringify(lastResults, null, 2), "application/json");
    return;
  }

  // CSV — union of all keys across items, in first-seen order
  const keys = [];
  lastResults.forEach(item => {
    Object.keys(item).forEach(k => { if (!keys.includes(k)) keys.push(k); });
  });
  const escapeCsv = v => `"${String(v ?? "").replace(/"/g, '""')}"`;
  const rows = [keys.join(",")];
  lastResults.forEach(item => {
    rows.push(keys.map(k => escapeCsv(item[k])).join(","));
  });
  downloadFile("results.csv", rows.join("\n"), "text/csv");
}

// ── Render results — works for jobs AND any other structured data ──
function renderResults(items, goal) {
  const box = document.getElementById("resultsBox");
  if (!items || items.length === 0) {
    box.innerHTML = '<p class="empty">No results found.</p>';
    return;
  }

  // detect if it looks like job results
  const isJob = items.some(r => r.role || r.company);

  if (isJob) {
    renderJobTable(items, box);
  } else {
    renderGenericCards(items, box);
  }
}

function confidenceBadge(confidence) {
  if (confidence === null || confidence === undefined) return "—";
  const pct = Math.round(confidence * 100);
  const tier = confidence >= 0.7 ? "high" : confidence >= 0.4 ? "mid" : "low";
  return `<span class="confidence-badge ${tier}">${pct}%</span>`;
}

function renderJobTable(jobs, box) {
  let html = `
    <table>
      <thead>
        <tr><th>#</th><th>Role</th><th>Company</th><th>Location</th><th>Salary</th><th>Confidence</th><th>Apply</th></tr>
      </thead><tbody>`;
  jobs.forEach((job, i) => {
    const screenshotLink = job.screenshot_url
      ? `<a class="screenshot-link" href="${safeUrl(job.screenshot_url.startsWith("http") ? job.screenshot_url : API + job.screenshot_url)}" target="_blank">📷</a>`
      : "";
    html += `<tr>
      <td>${i + 1}</td>
      <td>${escapeHtml(job.role) || "—"}</td>
      <td>${escapeHtml(job.company) || "—"}</td>
      <td>${escapeHtml(job.location) || "—"}</td>
      <td>${escapeHtml(job.salary) || "—"}</td>
      <td>${confidenceBadge(job.confidence)}</td>
      <td>${job.apply_url ? `<a href="${safeUrl(job.apply_url)}" target="_blank">Apply →</a>` : "—"}${screenshotLink}</td>
    </tr>`;
  });
  html += `</tbody></table>`;
  box.innerHTML = html;
}

function renderGenericCards(items, box) {
  // general-purpose: render any key-value pairs as cards
  let html = `<div class="generic-results">`;
  items.forEach((item, i) => {
    html += `<div class="result-card">`;
    html += `<div class="result-card-num">#${i + 1}</div>`;
    for (const [key, val] of Object.entries(item)) {
      if (key === "source_url") continue;
      if (!val || val === "null") continue;
      const isUrl = typeof val === "string" && val.startsWith("http");
      html += `<div class="result-row">
        <span class="result-key">${escapeHtml(key)}</span>
        <span class="result-val">${isUrl ? `<a href="${safeUrl(val)}" target="_blank">${escapeHtml(val)}</a>` : escapeHtml(val)}</span>
      </div>`;
    }
    if (item.source_url) {
      html += `<div class="result-source"><a href="${safeUrl(item.source_url)}" target="_blank">Source →</a></div>`;
    }
    html += `</div>`;
  });
  html += `</div>`;
  box.innerHTML = html;
}

// ── Render search results ──
function renderSearchResults(results) {
  const box = document.getElementById("resultsBox");
  if (!results || results.length === 0) {
    box.innerHTML = '<p class="empty">No results found.</p>';
    return;
  }
  let html = `<div class="search-results">`;
  results.forEach((r, i) => {
    html += `<div class="search-item">
      <div class="search-title">${i + 1}. ${escapeHtml(r.title)}</div>
      <a href="${safeUrl(r.url)}" target="_blank">${escapeHtml(r.url)}</a>
      <div class="search-snippet">${escapeHtml(r.snippet)}</div>
    </div>`;
  });
  html += `</div>`;
  box.innerHTML = html;
}

// ── Raw output (browse) ──
function showRawOutput(text) {
  document.getElementById("toolOutputCard").hidden = false;
  document.getElementById("toolOutput").textContent = text;
}

// ── Run history ──
async function loadRunHistory() {
  const box = document.getElementById("historyList");
  try {
    const res = await fetch(`${API}/runs?limit=20`, { headers: apiHeaders() });
    if (!res.ok) throw new Error(`Server error: ${res.status}`);
    const data = await res.json();
    renderRunHistory(data.runs || []);
  } catch (err) {
    box.innerHTML = `<p class="empty">Could not load history: ${escapeHtml(err.message)}</p>`;
  }
}

function renderRunHistory(runs) {
  const box = document.getElementById("historyList");
  if (!runs.length) {
    box.innerHTML = '<p class="empty">No runs yet.</p>';
    return;
  }
  box.innerHTML = runs.map(r => {
    const statusClass = r.status === "done" ? "status-done" : r.status === "error" ? "status-error" : "";
    const time = new Date(r.created_at).toLocaleString([], { hour12: false });
    return `<div class="history-item" onclick="loadRunDetail('${r.run_id}')">
      <div class="history-item-goal">${escapeHtml(r.goal)}</div>
      <div class="history-item-meta">
        <span class="${statusClass}">${escapeHtml(r.status)} · ${r.jobs_found} result${r.jobs_found === 1 ? "" : "s"}</span>
        <span>${escapeHtml(time)}</span>
      </div>
    </div>`;
  }).join("");
}

async function loadRunDetail(runId) {
  try {
    const res = await fetch(`${API}/runs/${runId}`, { headers: apiHeaders() });
    if (!res.ok) throw new Error(`Server error: ${res.status}`);
    const run = await res.json();

    clearLogs();
    resetRail();
    document.getElementById("toolOutputCard").hidden = true;
    document.getElementById("clarifyPanel").hidden = true;
    showToolTag("run_agent (history)");

    if (run.status === "error") setRailErrorAt("validator");
    else setRailAllDone();
    setRailMeta("browser", `${run.pages_visited} page${run.pages_visited === 1 ? "" : "s"}`);
    setRailMeta("validator", `${run.jobs_found} kept`);
    document.getElementById("railStatus").textContent = `history — ${run.status}`;

    log(`Loaded past run: ${run.goal}`, "info");
    renderResults(run.jobs, run.goal);
    setExportable(run.jobs);
  } catch (err) {
    log(`Error loading run: ${err.message}`, "error");
  }
}

// ── Enter key ──
document.addEventListener("DOMContentLoaded", () => {
  document.getElementById("goalInput")
    .addEventListener("keydown", e => { if (e.key === "Enter") runAuto(); });
  document.getElementById("clarifyInput")
    .addEventListener("keydown", e => { if (e.key === "Enter") answerClarify(); });
});

checkServer();
setInterval(checkServer, 10000);
loadRunHistory();
