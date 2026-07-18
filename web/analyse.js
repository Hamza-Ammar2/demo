/* Quick check — short scripted chat → cute small analysis */
const chat = document.getElementById("chat");
const actions = document.getElementById("actions");
const result = document.getElementById("result");
const resultBody = document.getElementById("result-body");

const answers = {};
const steps = [
  {
    id: "days_since_period",
    bot: "When was day 1 of your last period? Roughly is fine.",
    choices: [
      { label: "Within the last week", value: 4 },
      { label: "1–2 weeks ago", value: 10 },
      { label: "About 3 weeks ago", value: 21 },
      { label: "4+ weeks / unsure", value: 30 },
    ],
  },
  {
    id: "top_symptom",
    bot: "What’s bothering you most right now?",
    choices: [
      { label: "Headache / migraine", value: "headache" },
      { label: "Cramps", value: "cramps" },
      { label: "Fatigue / brain fog", value: "fatigue" },
      { label: "Mood swings", value: "mood" },
    ],
  },
  {
    id: "sleep",
    bot: "How has sleep been this week?",
    choices: [
      { label: "Pretty solid", value: "ok" },
      { label: "A bit rough", value: "rough" },
      { label: "Really bad (<6h a lot)", value: "bad" },
    ],
  },
  {
    id: "pill",
    bot: "Any contraception or medication change in the last 3 months?",
    choices: [
      { label: "No change", value: "none" },
      { label: "Yes — switched / started / stopped", value: "changed" },
      { label: "Not sure", value: "unsure" },
    ],
  },
];

let step = 0;

function addBubble(text, who) {
  const el = document.createElement("div");
  el.className = `bubble ${who}`;
  el.textContent = text;
  chat.appendChild(el);
  chat.scrollTop = chat.scrollHeight;
}

function showStep() {
  const s = steps[step];
  addBubble(s.bot, "bot");
  actions.innerHTML = "";
  s.choices.forEach((c) => {
    const b = document.createElement("button");
    b.type = "button";
    b.className = "chip-btn";
    b.textContent = c.label;
    b.onclick = () => {
      answers[s.id] = c.value;
      addBubble(c.label, "user");
      step += 1;
      actions.innerHTML = "";
      if (step < steps.length) showStep();
      else finish();
    };
    actions.appendChild(b);
  });
}

function finish() {
  addBubble("Okay — sketching a tiny read from that…", "bot");
  const d = answers.days_since_period;
  let windowGuess = "unclear";
  if (d <= 5) windowGuess = "menstrual-ish window";
  else if (d <= 13) windowGuess = "follicular-ish window";
  else if (d <= 16) windowGuess = "fertile-ish window";
  else if (d <= 28) windowGuess = "luteal-ish window";
  else windowGuess = "cycle timing unclear";

  const sleepOverlap =
    answers.sleep === "bad" || answers.sleep === "rough"
      ? "Sleep looks like a possible confounder for how you’re feeling."
      : "Sleep doesn’t jump out as the main confounder from this snapshot.";

  const completeness =
    answers.pill === "changed" ? 55 : answers.pill === "unsure" ? 45 : 62;
  const missing = [];
  if (answers.pill !== "changed") missing.push("exact contraceptive formulation (if any)");
  missing.push("more than one cycle of dated symptoms");
  if (answers.days_since_period >= 30) missing.push("confirmed period dates");

  const symptomNote = {
    headache: "Headaches often cluster near menstrual or late-luteal windows — worth dating a few episodes.",
    cramps: "Cramps near day 1 are common; track severity across two cycles to see a pattern.",
    fatigue: "Fatigue is noisy — sleep and stress usually compete with hormonal timing.",
    mood: "Mood shifts are multi-cause; a short timeline helps separate cycle timing from life stress.",
  }[answers.top_symptom];

  result.hidden = false;
  resultBody.innerHTML = `
    <div class="score-ring" style="--p:${completeness}%"><span>${completeness}%</span></div>
    <p class="muted" style="margin-top:-8px;font-size:0.85rem">Snapshot completeness (how much a fuller brief could use)</p>
    <p><strong>Likely cycle window:</strong> ${esc(windowGuess)}</p>
    <p>${esc(symptomNote)}</p>
    <p>${esc(sleepOverlap)}</p>
    <p><strong>One missing piece:</strong> ${esc(missing[0])}</p>
    <div class="chat-actions" style="margin-top:16px">
      <a class="btn" href="/analyse-full">Go deeper →</a>
    </div>
  `;
}

addBubble("Hi — this is the short look. Four quick taps, then a tiny pattern sketch.", "bot");
showStep();
