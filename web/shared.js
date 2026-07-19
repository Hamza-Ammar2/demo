/* shared helpers */
async function api(path, opts = {}) {
  const r = await fetch(path, {
    headers: { "Content-Type": "application/json", ...(opts.headers || {}) },
    ...opts,
  });
  if (!r.ok) {
    const t = await r.text();
    throw new Error(t || r.statusText);
  }
  return r.json();
}

function esc(s) {
  return String(s).replace(/[&<>]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" }[c]));
}

function toneClass(level) {
  if (level === "low") return "tone-low";
  if (level === "high") return "tone-high";
  return "tone-mid";
}

function renderBrief(container, d) {
  const b = d.brief;
  const bucket = (title, items, cls) => `
    <div class="panel" style="padding:14px">
      <div class="label">${esc(title)}</div>
      <ul class="muted" style="margin:0;padding-left:18px;font-size:0.92rem">
        ${(items.length ? items : ["(none)"]).map((x) => `<li>${esc(x)}</li>`).join("")}
      </ul>
    </div>`;

  container.innerHTML = `
    <div class="panel">
      <div class="label">Doctor brief · ${esc(b.analysis_mode)}</div>
      <p style="font-size:1.1rem;margin:0 0 8px">${esc(b.opening_statement)}</p>
    </div>
    <div class="cols-2">
      <div class="panel">
        <div class="label">Strongest findings</div>
        <ol style="margin:0;padding-left:18px">${b.strongest_findings.map((x) => `<li>${esc(x)}</li>`).join("")}</ol>
      </div>
      <div class="panel">
        <div class="label">Unresolved questions</div>
        <ol style="margin:0;padding-left:18px">${b.unresolved_questions.map((x) => `<li>${esc(x)}</li>`).join("")}</ol>
      </div>
    </div>
    <div class="cols-2">
      ${bucket("Established", b.established)}
      ${bucket("Possible", b.possible)}
      ${bucket("Not established", b.not_established)}
      ${bucket("Missing", b.missing)}
    </div>`;
}
