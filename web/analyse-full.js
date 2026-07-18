const results = document.getElementById("results");
const preds = document.getElementById("preds");
const phaseBars = document.getElementById("phase-bars");
const spark = document.getElementById("spark");
const brief = document.getElementById("brief");
const chatWrap = document.getElementById("chat-wrap");
const chat = document.getElementById("chat");
const actions = document.getElementById("actions");

const intake = {};

document.getElementById("btn-sarah").onclick = () => runFull({ source: "sarah" });
document.getElementById("btn-chat").onclick = () => {
  chatWrap.hidden = false;
  startChat();
};

function addBubble(text, who) {
  const el = document.createElement("div");
  el.className = `bubble ${who}`;
  el.textContent = text;
  chat.appendChild(el);
  chat.scrollTop = chat.scrollHeight;
}

function startChat() {
  chat.innerHTML = "";
  actions.innerHTML = "";
  const steps = [
    {
      id: "age",
      bot: "Rough age band?",
      choices: [
        { label: "Under 35", value: 30 },
        { label: "35–44", value: 40 },
        { label: "45–54", value: 50 },
        { label: "55+", value: 58 },
      ],
    },
    {
      id: "hot_flashes",
      bot: "Hot flashes or night sweats lately?",
      choices: [
        { label: "Rarely / never", value: 0 },
        { label: "Sometimes", value: 2 },
        { label: "Often", value: 4 },
      ],
    },
    {
      id: "pain_days",
      bot: "How many high-pain / flare-ish days in the last 2 weeks?",
      choices: [
        { label: "0–1", value: 1 },
        { label: "2–4", value: 3 },
        { label: "5+", value: 6 },
      ],
    },
    {
      id: "symptom_load",
      bot: "Overall symptom load this month?",
      choices: [
        { label: "Light", value: "light" },
        { label: "Medium", value: "medium" },
        { label: "Heavy", value: "heavy" },
      ],
    },
  ];
  let i = 0;
  const go = () => {
    const s = steps[i];
    addBubble(s.bot, "bot");
    actions.innerHTML = "";
    s.choices.forEach((c) => {
      const b = document.createElement("button");
      b.type = "button";
      b.className = "chip-btn";
      b.textContent = c.label;
      b.onclick = () => {
        intake[s.id] = c.value;
        addBubble(c.label, "user");
        i += 1;
        actions.innerHTML = "";
        if (i < steps.length) go();
        else runFull({ source: "intake", intake });
      };
      actions.appendChild(b);
    });
  };
  addBubble("We’ll still run the full Sarah-style engine demo underneath, and layer your answers onto the model chips.", "bot");
  go();
}

function predChip(name, value, level, note) {
  return `<div class="pred ${toneClass(level)}">
    <div class="name">${esc(name)}</div>
    <div class="value">${esc(value)}</div>
    <div class="note">${esc(note)}</div>
  </div>`;
}

async function runFull({ source, intake: answers = {} }) {
  results.hidden = false;
  brief.innerHTML = `<div class="panel muted">Compiling full brief…</div>`;
  preds.innerHTML = "";
  phaseBars.innerHTML = "";
  spark.innerHTML = "";

  try {
    const d = await api("/demo/sarah?mode=retrospective");

    // Hormonal state model (multi-source) — menstrual-leaning demo vector
    let phasePred = null;
    try {
      phasePred = await api("/models/hormonal-state", {
        method: "POST",
        body: JSON.stringify({
          features: {
            headaches_ord: 5, cramps_ord: 4, fatigue_ord: 4, sleepissue_ord: 4,
            moodswing_ord: 3, stress_ord: 3, bloating_ord: 3, sorebreasts_ord: 2,
            foodcravings_ord: 2, indigestion_ord: 2, appetite_ord: 3,
            sleep_minutes: 340, sleep_efficiency: 80, resting_hr: 76,
            steps_sum: 5200, stress_score_mean: 65, wrist_temp_delta: 0.05,
            glucose_mean: 98, hrv_rmssd: 28, is_weekend: 0,
          },
        }),
      });
    } catch (_) {
      phasePred = null;
    }

    // Menopause stage model
    const age = answers.age || 34;
    const hot = answers.hot_flashes ?? 0;
    let menoPred = null;
    try {
      menoPred = await api("/models/menopause-stage", {
        method: "POST",
        body: JSON.stringify({
          features: {
            age_years: age,
            fsh_miu_ml: age >= 50 ? 45 : age >= 45 ? 25 : 8,
            estradiol_pg_ml: age >= 50 ? 25 : 70,
            shbg_nmol_l: 50,
            hot_flash_freq: hot,
            night_sweat_freq: Math.max(0, hot - 1),
            sleep_disturbance: hot >= 2 ? 2 : 1,
            cycle_irregularity: age >= 45 ? 2 : 0,
            amenorrhea_months: age >= 55 ? 14 : age >= 50 ? 4 : 0,
            bmi: 26,
          },
        }),
      });
    } catch (_) {
      menoPred = null;
    }

    // Demo chips (honest framing)
    const menoStage = menoPred?.prediction?.predicted_stage || "premenopausal";
    let menoBefore45 = "Low";
    let menoTone = "low";
    if (age < 45 && (menoStage.includes("peri") || menoStage.includes("post"))) {
      menoBefore45 = "Elevated signal";
      menoTone = "high";
    } else if (age >= 45 && age < 50) {
      menoBefore45 = "Mid";
      menoTone = "mid";
    }

    const pain = answers.pain_days ?? 2;
    let endo = "Low";
    let endoTone = "low";
    if (pain >= 5) { endo = "Higher"; endoTone = "high"; }
    else if (pain >= 3) { endo = "Mid"; endoTone = "mid"; }

    const state = phasePred?.prediction?.predicted_state || "Luteal";
    const stateConf = phasePred?.prediction?.confidence;
    const stateLabel = stateConf != null ? `${state} (${Math.round(stateConf * 100)}%)` : state;

    preds.innerHTML =
      predChip("Menopause before 45 (signal)", menoBefore45, menoTone,
        "Stage-model + age heuristic — not a calendar prediction or diagnosis.") +
      predChip("Endo flare risk (demo)", endo, endoTone,
        "From recent high-pain days you reported. Not an endometriosis diagnosis.") +
      predChip("Likely hormonal state", stateLabel, "mid",
        "Multi-source model estimate from wearable/symptom-like features.");

    // Phase probability bars
    const probs = phasePred?.prediction?.probabilities || {
      Menstrual: 0.2, Follicular: 0.25, Fertility: 0.2, Luteal: 0.35,
    };
    phaseBars.innerHTML = Object.entries(probs).map(([k, v]) => `
      <div class="bar-row">
        <span>${esc(k)}</span>
        <div class="bar-track"><div class="bar-fill" data-w="${Math.round(v * 100)}"></div></div>
        <span class="mono">${Math.round(v * 100)}%</span>
      </div>`).join("");
    requestAnimationFrame(() => {
      phaseBars.querySelectorAll(".bar-fill").forEach((el) => {
        el.style.width = el.getAttribute("data-w") + "%";
      });
    });

    // Spark timeline
    const loads = answers.symptom_load === "heavy" ? [4, 6, 8, 7, 9, 6] :
      answers.symptom_load === "light" ? [2, 3, 2, 4, 3, 2] : [3, 5, 4, 6, 5, 4];
    spark.innerHTML = loads.map((n, idx) =>
      `<i class="${n >= 6 ? "on" : ""}" style="height:${n * 10}px;animation-delay:${idx * 0.05}s"></i>`
    ).join("");

    brief.innerHTML = "";
    renderBrief(brief, d);
  } catch (e) {
    brief.innerHTML = `<div class="panel" style="color:var(--rose)">${esc(e.message)}</div>`;
  }
}
