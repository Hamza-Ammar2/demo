const ta = document.getElementById("case-json");
const out = document.getElementById("out");

document.getElementById("btn-load-sarah").onclick = async () => {
  const d = await api("/demo/sarah?mode=retrospective");
  // Show a compact case-shaped preview: findings count + note to use demo compile
  ta.value = JSON.stringify({
    _note: "Sarah is preloaded server-side. Click Compile to run /demo/sarah, or paste your own Case JSON for POST /cases/compile.",
    demo: "sarah",
    mode: "retrospective",
    preview_findings: d.findings.map((f) => f.finding_id),
  }, null, 2);
};

document.getElementById("btn-compile").onclick = async () => {
  let payload;
  try {
    payload = JSON.parse(ta.value || "{}");
  } catch {
    alert("JSON could not be parsed.");
    return;
  }
  out.hidden = false;
  out.innerHTML = `<div class="panel muted">Compiling…</div>`;
  try {
    let d;
    if (payload.demo === "sarah") {
      d = await api(`/demo/sarah?mode=${payload.mode || "retrospective"}`);
    } else if (payload.subject && payload.events) {
      d = await api("/cases/compile", {
        method: "POST",
        body: JSON.stringify({ case: payload, mode: payload.mode || "retrospective" }),
      });
    } else {
      throw new Error("Paste a full Case object (subject + events) or load the Sarah fixture.");
    }
    out.innerHTML = "";
    const wrap = document.createElement("div");
    out.appendChild(wrap);
    renderBrief(wrap, d);
  } catch (e) {
    out.innerHTML = `<div class="panel" style="color:var(--rose)">${esc(e.message)}</div>`;
  }
};
