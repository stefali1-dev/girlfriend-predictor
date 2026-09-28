"use strict";

// Answers use the model's own field names and codes; see sagemaker/form.py.
let units = "metric";
const money = (v) => "$" + v.toLocaleString("en-US") + (v >= 150000 ? "+" : "");
const height = (v) => {
  if (units === "metric") return `${v} cm`;
  const inches = Math.round(v / 2.54);
  return `${Math.floor(inches / 12)}′${inches % 12}″`;
};
const weight = (v) => (units === "metric" ? `${v} kg` : `${Math.round(v * 2.2046)} lb`);
const outOf7 = (v) => `${v} of 7`;

const FIELDS = [
  { key: "age", label: "Age", name: "Your age", type: "range", min: 18, max: 43, start: 28,
    required: true, basic: true, note: "The survey covers ages 18 to 43.", show: String },
  { key: "sex", label: "Sex", name: "Your sex", type: "choice", basic: true,
    options: [["male", "Man"], ["female", "Woman"]] },
  { key: "education", label: "Education", name: "Your education", type: "choice", basic: true,
    options: [["less_than_hs", "No diploma"], ["hs", "High school"], ["some_college", "Some college"],
              ["bachelor_plus", "Degree"]] },
  { key: "student", label: "Student right now", name: "Student right now", type: "choice", basic: true,
    options: [[true, "Yes"], [false, "No"]] },
  { key: "work", label: "Work", name: "Your work", type: "choice", basic: true, skipName: "work",
    options: [["none", "No job"], ["part_time", "Part-time"], ["full_time", "Full-time"]] },
  { key: "earnings", label: "Yearly pay, before tax, in US dollars", name: "Your pay", type: "range",
    min: 0, max: 150000, step: 1000, start: 40000, basic: true, show: money, skipName: "pay",
    hidden: (a) => a.work === "none" },
  { key: "height_cm", label: "Height", name: "Your height", type: "range", min: 140, max: 210,
    start: 170, show: height, skipName: "height" },
  { key: "weight_kg", label: "Weight", name: "Your weight", type: "range", min: 40, max: 150,
    start: 75, show: weight, skipName: "weight" },
  { key: "big5_extraversion", label: "How outgoing are you?", name: "How outgoing you are",
    type: "range", min: 1, max: 7, start: 4, show: outOf7, ends: ["Reserved", "Outgoing"],
    skipName: "personality" },
  { key: "big5_conscientiousness", label: "How organised?", name: "How organised you are",
    type: "range", min: 1, max: 7, start: 4, show: outOf7, ends: ["Disorganised", "Organised"],
    skipName: "personality" },
  { key: "big5_emotional_stability", label: "How calm?", name: "How calm you are",
    type: "range", min: 1, max: 7, start: 4, show: outOf7, ends: ["Easily upset", "Calm"],
    skipName: "personality" },
  { key: "attendance_worship", label: "Religious services", name: "Religious services",
    type: "choice", skipName: "religious services",
    options: [[1, "Never"], [3, "Sometimes"], [6, "Weekly"]] },
  { key: "nonresident_children", label: "Your children who live elsewhere",
    name: "Children living elsewhere", type: "choice", skipName: "children",
    options: [[0, "None"], [1, "One"], [2, "Two or more"]] },
  { key: "census_region", label: "Where in the US you live", name: "Where you live", type: "select",
    options: [["", "Skip, or outside the US"], ["northeast", "Northeast"], ["midwest", "Midwest"],
              ["south", "South"], ["west", "West"]] },
  { key: "race_ethnicity", label: "Race or ethnicity", name: "Race or ethnicity", type: "select",
    note: "Optional. Asked because the survey found large gaps between groups.",
    options: [["", "Prefer not to say"], ["black", "Black"], ["hispanic", "Hispanic"],
              ["other", "Another (e.g. white, Asian)"]] },
];
// Keys are the size words sagemaker/form.py SIZES sends.
const SIZE_WIDTH = { "a little": 16, some: 30, "a lot": 50 };

const answers = {};
const painters = [];
const reducedMotion = matchMedia("(prefers-reduced-motion: reduce)");
const $ = (id) => document.getElementById(id);
const RING = 2 * Math.PI * 70;

let seq = 0;
let shown = null;
let hasResult = false;
let lastData = null;
let debounce;

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

function display(field, value) {
  if (field.show) return field.show(value);
  return field.options.find(([v]) => v === value)[1];
}

function renderField(field) {
  const wrap = el("div", "field");
  const id = `f-${field.key}`;
  const head = el(field.type === "choice" ? "div" : "label", "label");
  head.id = `${id}-label`;
  if (field.type !== "choice") head.htmlFor = id;
  const title = el("span", "", field.label);
  const right = el("span");
  const value = el("span", "value");
  const skip = el("button", "skip", "skip");
  skip.type = "button";
  skip.setAttribute("aria-label", `Skip ${field.label}`);
  right.append(value, " ", skip);
  head.append(title, right);
  wrap.append(head);
  if (field.note) wrap.append(el("p", "note small muted", field.note));

  let control;
  let paint;
  if (field.type === "range") {
    control = el("input");
    Object.assign(control, { type: "range", id, min: field.min, max: field.max,
                             step: field.step || 1, value: field.start });
    const set = () => { answers[field.key] = Number(control.value); paint(); changed(); };
    control.addEventListener("input", set);
    // Tapping an untouched slider without moving it still counts as an answer.
    control.addEventListener("pointerup", () => { if (!(field.key in answers)) set(); });
    paint = () => {
      const v = Number(control.value);
      control.style.setProperty("--fill", `${((v - field.min) / (field.max - field.min)) * 100}%`);
      const given = field.key in answers;
      value.textContent = given ? field.show(v) : "skipped";
      wrap.classList.toggle("unset", !given);
      control.setAttribute("aria-valuetext", given ? field.show(v) : "skipped");
    };
    wrap.append(control);
    if (field.ends) {
      const ends = el("div", "ends");
      ends.append(el("span", "", field.ends[0]), el("span", "", field.ends[1]));
      wrap.append(ends);
    }
  } else if (field.type === "choice") {
    control = el("div", "seg");
    control.setAttribute("role", "radiogroup");
    control.setAttribute("aria-labelledby", head.id);
    control.style.setProperty("--n", field.options.length);
    const buttons = field.options.map(([v, label]) => {
      const button = el("button", "", label);
      button.type = "button";
      button.setAttribute("role", "radio");
      // Tapping the picked option again clears it: every choice is optional.
      button.addEventListener("click", () => {
        if (answers[field.key] === v) delete answers[field.key];
        else answers[field.key] = v;
        paint();
        changed();
      });
      control.append(button);
      return button;
    });
    paint = () => {
      const picked = field.options.findIndex(([v]) => v === answers[field.key]);
      buttons.forEach((b, i) => b.setAttribute("aria-checked", String(i === picked)));
      control.classList.toggle("picked", picked >= 0);
      if (picked >= 0) control.style.setProperty("--i", picked);
    };
    wrap.append(control);
  } else {
    control = el("select");
    control.id = id;
    for (const [v, label] of field.options) {
      const option = el("option", "", label);
      option.value = v;
      control.append(option);
    }
    control.addEventListener("change", () => {
      if (control.value) answers[field.key] = control.value;
      else delete answers[field.key];
      paint();
      changed();
    });
    paint = () => { control.value = answers[field.key] || ""; };
    wrap.append(control);
  }

  skip.hidden = true;
  skip.addEventListener("click", () => { delete answers[field.key]; paint(); changed(); });
  const paintAll = () => {
    paint();
    skip.hidden = field.required || field.type === "select" || !(field.key in answers);
    if (field.hidden) wrap.hidden = field.hidden(answers);
  };
  painters.push(paintAll);
  paintAll();
  return wrap;
}

function payload() {
  const out = {};
  for (const field of FIELDS) {
    if (field.key in answers && !(field.hidden && field.hidden(answers))) out[field.key] = answers[field.key];
  }
  return out;
}

function changed() {
  painters.forEach((paint) => paint());
  if (!hasResult) return;
  seq++; // a reply for the earlier answers is stale now
  clearTimeout(debounce);
  debounce = setTimeout(() => predict(false), 400);
}

function setStatus(text, retry) {
  const status = $("status");
  status.textContent = text;
  if (retry) {
    const button = el("button", "", "Try again");
    button.type = "button";
    button.addEventListener("click", () => predict(!hasResult));
    status.append(" ", button);
  }
}

async function predict(first) {
  const mine = ++seq;
  const result = $("result");
  if (first && result.hidden) {
    result.hidden = false;
    result.classList.add("enter");
    result.scrollIntoView({ behavior: reducedMotion.matches ? "auto" : "smooth", block: "start" });
  }
  $("show").disabled = true;
  document.querySelector(".score").classList.add("busy");
  setStatus(hasResult ? "Updating…" : "Asking the model…");
  const slow = setTimeout(() => {
    if (mine === seq) setStatus("Waking up the model. The first answer can take up to 30 seconds.");
  }, 2000);
  try {
    const response = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload()),
      signal: AbortSignal.timeout(90000),
    });
    const data = await response.json();
    if (mine !== seq) return;
    if (!response.ok) throw new Error();
    render(data);
  } catch {
    if (mine === seq) setStatus("The model didn’t answer.", true);
  } finally {
    clearTimeout(slow);
    if (mine === seq) {
      $("show").disabled = false;
      document.querySelector(".score").classList.remove("busy");
    }
  }
}

function animateNumber(from, to) {
  const draw = (v) => {
    $("percent").textContent = Math.round(v);
    document.querySelector(".score .fill").style.strokeDashoffset = RING * (1 - v / 100);
  };
  if (reducedMotion.matches) { draw(to); return; }
  const start = performance.now();
  const step = (now) => {
    const t = Math.min(1, (now - start) / 750);
    draw(from + (to - from) * (1 - (1 - t) ** 3));
    if (t < 1 && shown === to) requestAnimationFrame(step);
  };
  requestAnimationFrame(step);
}

function render(data) {
  lastData = data;
  const percent = Math.round(data.probability * 100);
  const average = Math.round(data.average * 100);
  const from = shown ?? average;
  shown = percent;
  animateNumber(from, percent);
  $("sentence").textContent =
    `Of 100 people like you in the survey, about ${percent} lived with a partner.`;
  $("average").textContent = `${average}%`;
  $("peek-value").textContent = `${percent}%`;
  renderRows(data);

  const skipped = [...new Set(data.assumed.map((key) => FIELDS.find((f) => f.key === key).skipName))];
  $("assumed").textContent = !skipped.length ? ""
    : skipped.length <= 3
      ? `You skipped ${joinWords(skipped)}, so the model used typical answers for your age and sex.`
      : `You skipped ${skipped.length} questions, so the model used typical answers for your age and sex.`;

  if (!hasResult) {
    hasResult = true;
    $("show").hidden = true;
    $("peek").hidden = false;
  }
  setStatus("Change any answer and the number follows.");
}

function renderRows(data) {
  const current = data.leaned_on.filter((row) => row.answer in answers);
  $("leaned").hidden = !current.length;
  const rows = $("rows");
  rows.replaceChildren(...current.map((row, i) => {
    const field = FIELDS.find((f) => f.key === row.answer);
    const li = el("li");
    li.style.animationDelay = `${i * 70}ms`;
    const text = el("div", "row-text");
    const verb = row.direction === "up" ? "pushed it up" : "pulled it down";
    text.append(el("span", "", `${field.name}: ${display(field, answers[field.key])}`),
                el("span", "", `${verb} ${row.size}`));
    const bar = el("div", "bar");
    const fill = el("span", row.direction);
    bar.append(fill);
    li.append(text, bar);
    requestAnimationFrame(() => requestAnimationFrame(() => { fill.style.width = `${SIZE_WIDTH[row.size]}%`; }));
    return li;
  }));
}

function joinWords(words) {
  return words.length === 1 ? words[0]
    : `${words.slice(0, -1).join(", ")} and ${words[words.length - 1]}`;
}

function setUnits(next) {
  units = next;
  document.querySelectorAll(".units button").forEach((b) =>
    b.setAttribute("aria-pressed", String(b.dataset.units === units)));
  painters.forEach((paint) => paint());
  if (lastData) renderRows(lastData);
}

function init() {
  answers.age = FIELDS[0].start;
  $("basics").append(...FIELDS.filter((f) => f.basic).map(renderField));
  $("extra").append(...FIELDS.filter((f) => !f.basic).map(renderField));
  document.querySelector(".score .fill").style.strokeDasharray = RING;
  document.querySelector(".score .fill").style.strokeDashoffset = RING;
  document.querySelectorAll(".units button").forEach((b) =>
    b.addEventListener("click", () => setUnits(b.dataset.units)));
  $("form").addEventListener("submit", (event) => { event.preventDefault(); predict(true); });

  const peek = $("peek");
  peek.addEventListener("click", () =>
    $("result").scrollIntoView({ behavior: reducedMotion.matches ? "auto" : "smooth" }));
  new IntersectionObserver(([entry]) => peek.classList.toggle("away", entry.isIntersecting))
    .observe($("result"));

  // Wake the model while the person fills in the form: a cold start can take ~30 seconds.
  fetch("/api/predict", { method: "POST", headers: { "Content-Type": "application/json" },
                          body: JSON.stringify({ age: 30 }) }).catch(() => {});
}

init();
