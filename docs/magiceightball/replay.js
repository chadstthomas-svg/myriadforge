(() => {
  const picker = document.getElementById("scenario");
  const status = document.getElementById("replay-status");
  const input = document.getElementById("case-input");
  const output = document.getElementById("case-output");
  const results = document.getElementById("case-decisions");
  const headings = [["Expected", "expected"], ["CPU", "cpu"], ["Physical NCS1", "ncs1"]];

  function show(caseData) {
    input.textContent = caseData.text;
    results.replaceChildren(...headings.map(([label, key]) => {
      const card = document.createElement("div");
      const name = document.createElement("dt");
      const value = document.createElement("dd");
      const payload = document.createElement("pre");
      const { answers, category, action } = caseData[key];
      name.textContent = label;
      payload.textContent = JSON.stringify({ may_route: answers.may_route, category, action }, null, 2);
      value.append(payload);
      card.append(name, value);
      return card;
    }));
    output.hidden = false;
  }

  async function load() {
    try {
      const response = await fetch("./frontline-cases.json");
      if (!response.ok) throw new Error("Cases unavailable");
      const cases = await response.json();
      for (const [index, item] of cases.entries()) picker.add(new Option(item.title, String(index)));
      picker.addEventListener("change", () => show(cases[Number(picker.value)]));
      show(cases[0]);
      picker.disabled = false;
      status.textContent = "";
    } catch {
      status.textContent = "Replay could not load. Open the CPU or NCS1 report to read the results.";
    }
  }

  load();
})();
