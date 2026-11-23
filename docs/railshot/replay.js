(() => {
  const get = id => document.getElementById(id);
  const bank = get("bank"), backend = get("backend"), next = get("next"), reset = get("reset");
  const status = get("replay-status"), rows = get("requests");
  const eventNames = [
    "Open stream", "Reject sequence gap", "Complete approval", "Independent stream",
    "Restart and retry", "Apply veto", "Reject conflicting retry", "Reset and approve",
  ];
  let events, index;

  function advance() {
    const { request, response, process } = events[index];
    const row = document.createElement("tr");
    const name = document.createElement("th");
    name.scope = "row";
    name.textContent = `Process ${process} · ${eventNames[index]}`;
    row.append(name);
    const result = response.ok
      ? response.decision.result.action ?? "No proposal"
      : response.error.code;
    for (const text of [request.stream_id + " · seq " + request.seq, request.append, result]) {
      const cell = document.createElement("td");
      cell.textContent = text;
      row.append(cell);
    }
    rows.append(row);
    index++;
    status.textContent = `${index} of ${events.length} recorded requests`;
    next.disabled = index === events.length;
  }

  function setup(run) {
    events = [run.process_1, run.process_2].flatMap((phase, process) =>
      phase.requests.map((request, position) => ({ request, response: phase.responses[position], process: process + 1 }))
    );
    rows.replaceChildren();
    index = 0;
    advance();
  }

  async function load() {
    try {
      const response = await fetch("./industry-results.json");
      if (!response.ok) throw new Error("Results unavailable");
      const { ranges } = await response.json();
      const show = () => setup(ranges.find(run => run.bank === bank.value && run.backend === backend.value));
      bank.addEventListener("change", show);
      backend.addEventListener("change", show);
      reset.addEventListener("click", show);
      next.addEventListener("click", advance);
      show();
      bank.disabled = false;
      backend.disabled = false;
      reset.disabled = false;
      get("replay-log").hidden = false;
    } catch {
      status.textContent = "Replay could not load. Open the labeled results to see the recorded requests.";
    }
  }

  load();
})();
