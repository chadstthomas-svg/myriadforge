(() => {
  if (new URLSearchParams(location.search).get("embed") === "1") {
    document.body.classList.add("embed");
    const main = document.querySelector("main");
    new ResizeObserver(() => {
      const height = Math.ceil(main.getBoundingClientRect().bottom) + 12;
      window.parent.postMessage({type:"racklab:bankshot-height",height}, location.origin);
    }).observe(main);
  }
  const get = (id) => document.getElementById(id);
  const form = get("search-form"), note = get("query"), domain = get("domain"), example = get("example");
  const status = get("status"), results = get("results"), expected = get("expectation"), submit = get("submit");
  const vehicleFields = ["year","make","model"].map(get);
  const cases = new Map();
  let worker;
  function runSearch(query) {
    worker ??= new Worker("./search-worker.js");
    return new Promise((resolve,reject) => {
      worker.onmessage = ({data}) => data.error ? reject(new Error(data.error)) : resolve(data.result);
      worker.onerror = () => {
        worker.terminate();
        worker = undefined;
        reject(new Error("Browser search could not start. Open Field Notes for replay results."));
      };
      worker.postMessage(query);
    });
  }
  function syncVehicle() { get("vehicle").hidden = domain.value !== "dealership"; }
  function clearCase() {
    example.value = "";
    expected.hidden = true;
    syncVehicle();
  }
  async function loadExamples() {
    try {
      const response = await fetch("./field-notes/cases.json");
      if (!response.ok) return;
      for (const item of await response.json()) {
        cases.set(item.id, item);
        example.append(new Option(item.title,item.id));
      }
    } catch { /* Custom searches remain available. */ }
  }
  example.addEventListener("change", () => {
    const item = cases.get(example.value);
    if (!item) return clearCase();
    note.value = item.note;
    domain.value = item.domain;
    vehicleFields.forEach((field,index) => { field.value = item.vehicle?.[index] ?? ""; });
    syncVehicle();
    expected.textContent = "Expected reference · " + item.reference;
    expected.hidden = false;
  });
  for (const control of [note,domain,...vehicleFields]) control.addEventListener("input", clearCase);
  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    const item = cases.get(example.value);
    const values = vehicleFields.map(field => field.value.trim());
    const wantsVehicle = domain.value === "dealership" && values.some(Boolean);
    if (wantsVehicle && !values.every(Boolean)) {
      status.textContent = "Enter the year, make, and model, or clear all vehicle fields.";
      return;
    }
    const vehicle = wantsVehicle ? [Number(values[0]),values[1],values[2]] : null;
    results.replaceChildren();
    status.textContent = "Loading or running search in your browser…";
    submit.disabled = true;
    try {
      const data = await runSearch({domain:domain.value,text:note.value,vehicle});
      const rank = item ? data.hits.findIndex(hit => hit.id === item.expected) + 1 : null;
      const label = ({needs_details:"Add a case note or vehicle details",no_match:"No matching reference",ambiguous:"Multiple possible references",unsupported:"Catalog unavailable"})[data.status];
      status.textContent = (label || data.hits.length + " references") + " · browser " + data.request_ms.toFixed(1) + " ms"
        + (item ? " · " + (rank ? "Expected source at rank " + rank : "Expected source outside top 10") : "");
      for (const hit of data.hits) {
        const row = document.createElement("li");
        if (item?.expected === hit.id) row.dataset.target = "";
        const link = document.createElement("a");
        link.href = hit.url; link.target = "_blank"; link.rel = "noopener"; link.textContent = hit.title;
        row.append(link); results.append(row);
      }
    } catch (error) {
      status.textContent = error.message;
    } finally {
      submit.disabled = false;
    }
  });
  loadExamples();
})();
