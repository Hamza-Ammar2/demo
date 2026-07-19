/* "I've been feeling off" — data-driven conversational intake.
   Questions come from /analyse/question-plan (ordered by real model importances).
   The read is grounded in mcPHASES cohort base-rates + the menopause-stage model. */

const chat = document.getElementById("chat");
const quick = document.getElementById("quick");
const notedEl = document.getElementById("noted");
const nextQ = document.getElementById("next-q");
const whyEl = document.getElementById("why");
const nextBox = document.getElementById("next-box");
const softRead = document.getElementById("soft-read");
const softBody = document.getElementById("soft-body");
const groundingLine = document.getElementById("grounding");
const layout = document.getElementById("layout");
const intakeDone = document.getElementById("intake-done");

const SYMPTOM_NAMES = {
  migraine: "migraine", headache: "headaches", cramps: "cramps",
  pelvic_pain: "pelvic pain", fatigue: "fatigue", brain_fog: "brain fog",
  mood: "mood changes", bloating: "bloating", sore_breasts: "breast tenderness",
  nausea: "nausea",
};

// how each question id writes into the intake object
const APPLY = {
  symptoms: (v) => addSymptom(v),
  severity: (v) => { intake.symptoms = intake.symptoms.map((s) => ({ ...s, severity: v })); note(`Intensity: ${v}`); },
  symptom_timing: (v, l) => { intake.symptom_timing = v; note(`Timing: ${l.toLowerCase()}`); },
  last_period: (v) => { intake.last_period_days_ago = v; note(v === null ? "Last period: not sure" : `Last period: ~${v} days ago`); },
  cycle_regularity: (v, l) => { intake.cycle_regularity = v; note(`Cycles: ${l.toLowerCase()}`); },
  sleep: (v, l) => { intake.sleep_quality = v; note(`Sleep: ${l.toLowerCase()}`); },
  hot_flashes: (v, l) => { intake.hot_flash_freq = v; note(`Hot flashes: ${l.toLowerCase()}`); },
  night_sweats: (v, l) => { intake.night_sweat_freq = v; note(`Night sweats: ${l.toLowerCase()}`); },
  contraception_status: (v, l) => { intake.contraception_status = v; note(`Contraception: ${l.toLowerCase()}`); },
  contraception_type: (v, l) => { intake.contraception_formulation = l; note(`Type: ${l}`); },
  contraception_when: (v, l) => { intake.contraception_changed_days_ago = v; note(`Change: ${l.toLowerCase()}`); },
  age: (v, l) => { intake.age_range = v; note(`Age: ${l}`); },
  bloodwork: (v, l) => { intake.bloodwork = v; if (v !== "none") note(`Bloodwork: ${l.toLowerCase()}`); },
  other_changes: (v, l) => { addOther(v, l); },
};

const intake = { symptoms: [] };
const notes = [];
const answered = new Set();
let QUESTIONS = [];
let currentId = null;

/* Fallback question set so the page ALWAYS works, even if the data-driven
   /analyse/question-plan endpoint is unavailable (e.g. server not restarted). */
const DEFAULT_QUESTIONS = [
  { id: "symptoms", mode: "multi", prompt: "What have you been feeling? Tap all that fit, then Continue.",
    chips: [["Headaches / migraine", "headache"], ["Cramps", "cramps"], ["Pelvic pain", "pelvic_pain"],
      ["Fatigue", "fatigue"], ["Brain fog", "brain_fog"], ["Mood changes", "mood"],
      ["Bloating", "bloating"], ["Sore breasts", "sore_breasts"], ["Nausea", "nausea"]], maps_to: ["engine"] },
  { id: "severity", mode: "single", prompt: "At their worst, how intense?",
    chips: [["Mild", "mild"], ["Moderate", "moderate"], ["Severe", "severe"]], requires_symptoms: true, maps_to: ["engine"] },
  { id: "symptom_timing", mode: "single", prompt: "When do they tend to hit?",
    chips: [["Right before my period", "before_period"], ["During my period", "during_period"],
      ["After it ends", "after_period"], ["Mid-cycle / around ovulation", "mid_cycle"],
      ["All the time / no pattern", "constant"]], requires_symptoms: true, maps_to: ["engine"] },
  { id: "last_period", mode: "single", prompt: "When did your last period start?",
    chips: [["In the last few days", 3], ["1–2 weeks ago", 10], ["About 3 weeks ago", 21],
      ["4–8 weeks ago", 42], ["Over 2 months ago", 90], ["Not sure", null]], maps_to: ["engine"] },
  { id: "sleep", mode: "single", prompt: "How's your sleep been this past week?",
    chips: [["Fine", "ok"], ["A bit rough", "rough"], ["Really bad", "bad"]], maps_to: ["engine"] },
  { id: "cycle_regularity", mode: "single", prompt: "Are your cycles usually regular?",
    chips: [["Regular, ~monthly", "regular"], ["A bit irregular", "irregular"],
      ["Very unpredictable", "very_irregular"], ["No periods right now", "none"]], maps_to: [] },
  { id: "hot_flashes", mode: "single", prompt: "Any hot flashes lately?",
    chips: [["None", 0], ["Occasionally", 2], ["Often", 4], ["Many a day", 6]], maps_to: [] },
  { id: "night_sweats", mode: "single", prompt: "Night sweats?",
    chips: [["None", 0], ["Sometimes", 2], ["Frequently", 4]], maps_to: [] },
  { id: "contraception_status", mode: "single", prompt: "Any contraception right now?",
    chips: [["None", "none"], ["Yes, no recent changes", "on_stable"], ["Changed it recently", "changed"],
      ["Recently stopped", "stopped"]], maps_to: ["engine"] },
  { id: "contraception_type", mode: "single", prompt: "Which kind is it (or was it)?",
    chips: [["Combined pill", "combined_pill"], ["Progestogen-only pill", "pop"], ["Hormonal IUD", "hormonal_iud"],
      ["Copper IUD", "copper_iud"], ["Implant", "implant"], ["Injection", "injection"], ["Ring / patch", "ring_patch"],
      ["Not sure", "unknown"]], requires: { contraception_status: ["on_stable", "changed", "stopped"] }, maps_to: ["engine"] },
  { id: "contraception_when", mode: "single", prompt: "Roughly when did that change?",
    chips: [["In the last month", 25], ["1–3 months ago", 60], ["3–6 months ago", 135], ["Over 6 months ago", 220]],
    requires: { contraception_status: ["changed", "stopped"] }, maps_to: ["engine"] },
  { id: "age", mode: "single", prompt: "Which age range are you in?",
    chips: [["Under 20", "under-20"], ["20s", "20-29"], ["30s", "30-39"], ["40s", "40-49"], ["50+", "50-59"]], maps_to: [] },
  { id: "other_changes", mode: "multi", prompt: "Anything else changed lately? Tap any — or skip.",
    chips: [["New medication", "med"], ["Lots of stress", "stress"], ["Weight change", "weight"],
      ["New exercise", "exercise"], ["Big diet change", "diet"], ["Nothing else", "none"]], maps_to: ["engine"] },
];

function addSymptom(v) {
  if (!intake.symptoms.some((s) => s.type === v)) {
    intake.symptoms.push({ type: v, severity: "moderate", days_ago: null });
    note(`Symptom: ${SYMPTOM_NAMES[v] || v}`);
  }
}
function removeSymptom(v) {
  intake.symptoms = intake.symptoms.filter((s) => s.type !== v);
  const line = `Symptom: ${SYMPTOM_NAMES[v] || v}`;
  const i = notes.indexOf(line);
  if (i >= 0) { notes.splice(i, 1); renderNoted(); }
}
function addOther(v, l) {
  intake.other_changes = intake.other_changes || [];
  if (v !== "none" && !intake.other_changes.includes(v)) { intake.other_changes.push(v); note(`Also: ${l.toLowerCase()}`); }
}

function addBubble(text, who) {
  const el = document.createElement("div");
  el.className = `bubble ${who}`;
  el.textContent = text;
  chat.appendChild(el);
  chat.scrollTop = chat.scrollHeight;
}
function titleCaseNote(s) {
  if (!s) return s;
  // Keep short tokens like "FSH", "IUD", "PCOS" uppercase-ish if already acronym-looking
  return s.replace(/(^|[\s(/])([a-z])/g, (_, a, b) => a + b.toUpperCase());
}

function renderNoted() {
  if (!notes.length) {
    notedEl.innerHTML = `<li class="muted">Nothing yet — tap a chip to begin.</li>`;
    return;
  }
  notedEl.innerHTML = notes.map((n, idx) => {
    const i = n.indexOf(":");
    const inner = i > 0
      ? `<span class="noted-key">${esc(n.slice(0, i).trim())}</span><span class="noted-val">${esc(titleCaseNote(n.slice(i + 1).trim()))}</span>`
      : `<span class="noted-val">${esc(titleCaseNote(n))}</span>`;
    return `<li class="noted-chip" tabindex="0" data-i="${idx}" style="--i:${idx}">${inner}</li>`;
  }).join("");
  wireNotedHover();
}

function wireNotedHover() {
  const chips = [...notedEl.querySelectorAll(".noted-chip")];
  chips.forEach((chip) => {
    const on = () => {
      notedEl.classList.add("is-focusing");
      chips.forEach((c) => c.classList.toggle("is-hot", c === chip));
    };
    const off = () => {
      notedEl.classList.remove("is-focusing");
      chips.forEach((c) => c.classList.remove("is-hot"));
    };
    chip.addEventListener("mouseenter", on);
    chip.addEventListener("mouseleave", off);
    chip.addEventListener("focus", on);
    chip.addEventListener("blur", off);
  });
}
function note(line) { if (!notes.includes(line)) { notes.push(line); renderNoted(); } }

function requiresMet(q) {
  if (q.requires_symptoms && intake.symptoms.length === 0) return false;
  if (q.requires) {
    for (const [k, vals] of Object.entries(q.requires)) {
      if (!vals.includes(intake[k])) return false;
    }
  }
  return true;
}
function nextQuestion() {
  return QUESTIONS.find((q) => !answered.has(q.id) && requiresMet(q));
}

function ask() {
  const q = nextQuestion();
  quick.innerHTML = "";
  if (!q) {
    nextBox.hidden = true;
    layout.classList.add("results-mode");
    if (intakeDone) intakeDone.hidden = false;
    updateRead(true);
    return;
  }
  layout.classList.remove("results-mode");
  if (intakeDone) intakeDone.hidden = true;
  if (q.id !== currentId) {
    currentId = q.id;
    // Only the current question stays visible once the previous one is answered.
    chat.innerHTML = "";
    if (q.id === "symptoms") {
      addBubble("Hi — I'm here to help you make sense of how you've been feeling.", "bot");
    }
    addBubble(q.prompt, "bot");
    if (q.mode === "multi") addBubble("Tap all that apply, then press Continue →", "bot");
  }
  nextQ.textContent = q.prompt;
  whyEl.textContent = q.why
    ? `Why we ask: ${q.why}`
    : (q.importance > 0 ? `Feeds the menopause-stage model (importance ${q.importance}).` : "");

  const isChosen = (value) =>
    (q.id === "symptoms" && intake.symptoms.some((s) => s.type === value)) ||
    (q.id === "other_changes" && (intake.other_changes || []).includes(value));

  const renderContinue = () => {
    if (q.mode !== "multi") return;
    const n = q.id === "symptoms" ? intake.symptoms.length : (intake.other_changes || []).length;
    const cont = document.createElement("button");
    cont.type = "button";
    cont.className = "chip-btn primary";
    cont.textContent = n > 0 ? `Continue with ${n} selected →` : "Continue / skip →";
    cont.onclick = () => { answered.add(q.id); ask(); updateRead(); };
    quick.appendChild(cont);
  };

  q.chips.forEach(([label, value]) => {
    const b = document.createElement("button");
    b.type = "button"; b.className = "chip-btn"; b.textContent = label;
    if (q.mode === "multi" && isChosen(value)) b.classList.add("active");
    b.onclick = () => {
      if (q.mode === "multi") {
        if (isChosen(value)) {
          b.classList.remove("active");
          if (q.id === "symptoms") removeSymptom(value);
          else if (q.id === "other_changes") {
            intake.other_changes = (intake.other_changes || []).filter((x) => x !== value);
          }
        } else {
          b.classList.add("active");
          (APPLY[q.id] || (() => {}))(value, label);
        }
        const old = quick.querySelector(".chip-btn.primary");
        if (old) old.remove();
        renderContinue();
        updateRead();
      } else {
        (APPLY[q.id] || (() => {}))(value, label);
        answered.add(q.id);
        ask(); updateRead();
      }
    };
    quick.appendChild(b);
  });

  renderContinue();
}

/* ---- grounded auto read ---- */
function ready() { return intake.symptoms.length > 0 && answered.has("last_period"); }
let readInFlight = false;
async function updateRead(force = false) {
  if (!intake.symptoms.length) return;
  if (!ready() && !force) return;
  if (readInFlight) return;
  readInFlight = true;
  softRead.hidden = false;
  try {
    const d = await api("/analyse/feeling-off", {
      method: "POST",
      body: JSON.stringify({ intake, free_text: null, use_llm: true, consent: false }),
    });
    renderRead(d);
  } catch (e) {
    softBody.innerHTML = `<p style="color:var(--rose)">${esc(e.message)}</p>`;
  } finally { readInFlight = false; }
}

function renderRead(d) {
  const b = d.brief;
  const uniq = (arr) => {
    const seen = new Set();
    const out = [];
    for (const x of arr || []) {
      const k = String(x).trim().toLowerCase();
      if (!k || seen.has(k)) continue;
      seen.add(k);
      out.push(x);
    }
    return out;
  };
  const list = (arr) => {
    const items = uniq(arr);
    return items.length
      ? `<ul style="margin:6px 0 0;padding-left:18px">${items.map((x) => `<li>${esc(x)}</li>`).join("")}</ul>`
      : `<p class="muted" style="margin:6px 0 0">Nothing clear yet — add a little more.</p>`;
  };

  const f = d.foundation || {};
  const cards = f.cards || d.fundamentals || [];
  let foundationHtml = "";
  if (cards.length) {
    // Only these count as "what datasets show" — not guideline seeds or MedQuAD phrasing citations.
    const DATASET_EV = new Set(["mcphases", "nhanes", "pcos_kaggle", "swan"]);
    foundationHtml = `<div class="label" style="margin-top:16px">Medical foundation</div>
      <p class="muted" style="font-size:0.8rem;margin:4px 0 10px">Seeded clinical associations. Dataset numbers appear only when we actually have them.</p>
      <div class="fund-grid">` +
      cards.map((h) => {
        const fact = h.foundation_fact || h.talking_point || "";
        const realDs = (h.datasets || []).filter((d) => DATASET_EV.has(d));
        // Prefer evidence lines that look like real stats; hide bare "Guideline/textbook seed:" as fake "data"
        const evLines = (h.evidence_summaries || []).filter((s) => {
          const low = s.toLowerCase();
          if (low.startsWith("guideline/textbook seed")) return false;
          if (low.startsWith("related nih-sourced q&a")) return false;
          return true;
        });
        const ev = evLines.map((s) =>
          `<div class="cohort-line sig"><strong>From data:</strong> ${esc(s)}</div>`).join("");
        const personal = h.personal_pattern
          ? `<div class="muted" style="font-size:0.85rem;margin-top:6px"><em>Your pattern:</em> ${esc(h.personal_pattern)}</div>`
          : "";
        const ds = realDs.length
          ? ` · measured evidence from: ${realDs.join(", ")}`
          : ` · foundation seed only (no cohort/model numbers on this card)`;
        return `
        <div class="fund-card" tabindex="0" role="button" aria-expanded="false">
          <div class="fund-title">${esc(h.title)}</div>
          <div class="fund-detail">
            <div class="fund-detail-inner">
              <div class="fund-body">${esc(fact)}</div>
              ${ev || `<div class="muted" style="font-size:0.78rem;margin-top:8px">No dataset statistic attached to this edge yet.</div>`}
              ${personal}
              <div class="muted" style="font-size:0.75rem;margin-top:8px">Seed source: ${esc(h.source || "")}${esc(ds)}</div>
            </div>
          </div>
        </div>`;
      }).join("") + `</div>`;
  }

  let models = "";
  // Product models only — sequence research lives behind Research depth.
  const signals = (f.model_signals || [])
    .filter((m) => m && m.task !== "sequence_research")
    .concat(
      d.menopause_model && !(f.model_signals || []).some((x) => x.task === "menopause_stage") ? [d.menopause_model] : [],
      d.pcos_model && !(f.model_signals || []).some((x) => x.task === "pcos_risk") ? [d.pcos_model] : [],
    );
  if (signals.length) {
    models = `<div class="label" style="margin-top:16px">Model signals</div>` +
      signals.map((m) => {
        const title = m.task === "pcos_risk" ? "PCOS-risk model"
          : m.task === "menopause_stage" ? "Menopause-stage model"
          : (m.task || "Model");
        const detail = m.predicted_stage
          ? `${m.predicted_stage.replace(/_/g, " ")} (${Math.round((m.confidence || 0) * 100)}%)`
          : (m.probability != null ? `P≈${m.probability}` : "");
        return `
        <div class="model-chip">
          <div class="mc-stage">${esc(title)}</div>
          <div class="mc-conf">${esc(detail)}</div>
          <div class="muted" style="font-size:0.82rem;margin-top:6px">${esc(m.statement || "")}</div>
        </div>`;
      }).join("");
  }

  softBody.innerHTML = `
    <div style="font-size:1.02rem;margin-bottom:12px">${esc(b.opening_statement)}</div>
    <div class="label" style="margin-top:14px">Your patterns</div>
    ${list(b.strongest_findings)}
    ${foundationHtml}
    ${models}
    <div class="label" style="margin-top:16px">Bring / confirm these</div>
    ${list(b.missing)}
    <div class="label" style="margin-top:16px">Questions to ask Doctor</div>
    ${list(b.unresolved_questions)}`;
  wireFundCards();
  wireResearchDepth(d.research_depth_available !== false);
}

function wireFundCards() {
  softBody.querySelectorAll(".fund-card").forEach((card) => {
    const setOpen = (open) => {
      card.classList.toggle("is-open", open);
      card.setAttribute("aria-expanded", open ? "true" : "false");
    };
    // Click pins open (handy while reading); hover also expands via CSS
    card.addEventListener("click", () => setOpen(!card.classList.contains("is-open")));
    card.addEventListener("keydown", (e) => {
      if (e.key === "Enter" || e.key === " ") {
        e.preventDefault();
        setOpen(!card.classList.contains("is-open"));
      }
    });
  });
}

/* ---- Research depth (opt-in Track 02 panel; not part of soft read) ---- */
let researchDepthLoaded = false;
function wireResearchDepth(available) {
  const wrap = document.getElementById("research-depth");
  const btn = document.getElementById("research-depth-btn");
  const panel = document.getElementById("research-depth-panel");
  if (!wrap || !btn || !panel) return;
  wrap.hidden = false;
  if (available === false) {
    btn.disabled = true;
    btn.textContent = "Research depth · run make pfl-smoke to enable";
    return;
  }
  btn.disabled = false;
  btn.textContent = "Research depth · privacy-preserving sequence model";
  if (btn.dataset.wired === "1") return;
  btn.dataset.wired = "1";
  btn.addEventListener("click", async () => {
    const open = panel.hidden;
    panel.hidden = !open;
    btn.setAttribute("aria-expanded", open ? "true" : "false");
    wrap.classList.toggle("is-open", open);
    if (!open || researchDepthLoaded) return;
    panel.innerHTML = `<p class="muted" style="margin:0">Loading research…</p>`;
    try {
      const d = await api("/research/depth");
      researchDepthLoaded = true;
      panel.innerHTML = renderResearchDepth(d);
    } catch (e) {
      panel.innerHTML = `<p class="muted" style="margin:0">${esc(e.message)}</p>`;
    }
  });
}

function renderResearchDepth(d) {
  const fmt = (v) => (v == null || Number.isNaN(v) ? "—" : Number(v).toFixed(2));
  const rows = (d.comparison || []).map((r) => `
    <tr>
      <td>${esc(r.label)}</td>
      <td>${fmt(r.accuracy)}</td>
      <td>${fmt(r.f1)}</td>
      <td>${fmt(r.sensitivity)}</td>
      <td>${fmt(r.specificity)}</td>
    </tr>`).join("");
  return `
    <p class="research-depth-eyebrow">${esc(d.eyebrow || "Research depth")}</p>
    <h3 class="research-depth-title">${esc(d.title || "Personalized sequence (research)")}</h3>
    <p class="research-depth-lede">${esc(d.lede || "")}</p>
    <p class="muted" style="font-size:0.8rem;margin:8px 0 12px">${esc(d.needs_personal || "")}</p>
    <div class="muted" style="font-size:0.78rem;margin-bottom:10px">
      ${esc(String(d.n_clients ?? "?"))} clients · ${esc(String(d.rounds ?? "?"))} rounds
      ${d.checkpoint_ready ? " · checkpoint ready" : ""}
    </div>
    <table class="research-depth-table">
      <thead>
        <tr><th>Approach</th><th>Acc</th><th>F1</th><th>Sens</th><th>Spec</th></tr>
      </thead>
      <tbody>${rows}</tbody>
    </table>
    <p class="muted" style="font-size:0.75rem;margin:12px 0 0">${esc(d.honesty_note || "")}</p>
  `;
}

/* ---- boot: load the data-driven plan ---- */
(async function boot() {
  renderNoted();
  try {
    const plan = await api("/analyse/question-plan");
    QUESTIONS = (plan.questions && plan.questions.length) ? plan.questions : DEFAULT_QUESTIONS;
    // Soften any server prompts that still mention typing
    QUESTIONS = QUESTIONS.map((q) => ({
      ...q,
      prompt: String(q.prompt || "")
        .replace(/\s*[—–-]\s*or type your own\.?/gi, ".")
        .replace(/\s*or type your own\.?/gi, "")
        .replace(/Type anything[^.]*\./gi, "")
        .replace(/or tell me anything else first\??/gi, "")
        .trim(),
      choices: (q.choices || []).filter((c) => !/add more below/i.test(c)),
    }));
    const g = plan.grounding || {};
    const src = (g.mcphases || {});
    groundingLine.textContent =
      `Grounded in mcPHASES (n=${src.n_participants || "?"} participants) + a menopause-stage model. ` +
      `Questions are ordered by what the model finds most informative.`;
  } catch (e) {
    QUESTIONS = DEFAULT_QUESTIONS;
    groundingLine.textContent = "Grounded in cohort base-rates + a menopause-stage model.";
  }
  ask();
})();
