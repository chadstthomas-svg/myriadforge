(() => {
  const get = id => document.getElementById(id);
  const workflow = get("workflow"), condition = get("condition"), audio = get("clip");
  const status = get("test-status"), events = get("events"), cursor = get("cursor");
  const labels = { wake: "Wake start", interval: "Speech interval" };
  let seconds;

  function render(example, clip, sampleRate) {
    seconds = clip.samples / sampleRate;
    const percent = samples => Math.min(100, 100 * samples / clip.samples) + "%";
    const time = samples => (samples / sampleRate).toFixed(2) + " s";
    const wakeLabel = value => value ? "wake" : "quiet";
    const match = clip.expected_wake === clip.observed_wake;
    status.dataset.missed = String(!match);
    status.textContent = `Expected ${wakeLabel(clip.expected_wake)} · Recorded ${wakeLabel(clip.observed_wake)}`
      + (match ? "" : " · Missed wake");
    get("prompt").textContent = clip.condition === "ambient" ? "Ambient sound only." : example.input;
    get("ambient-credit").href = "https://freesound.org/s/" + example.ambient_source + "/";
    get("ambient-credit").textContent = "Freesound #" + example.ambient_source;
    audio.pause();
    audio.src = "./assets/" + clip.audio;
    cursor.style.left = "0%";
    // Paint interval spans first, then wake markers so a marker remains visible at the interval start.
    events.replaceChildren(...["interval", "wake"].flatMap(type =>
      clip.events.filter(event => event.type === type).map(event => {
        const segment = document.createElement("span");
        segment.className = event.type;
        segment.style.left = percent(event.start);
        if (event.type === "interval") segment.style.width = percent(event.end - event.start);
        return segment;
      })
    ));
    const descriptions = clip.events.map(event => {
      const start = labels[event.type] + " · " + time(event.start);
      return event.type === "wake" ? start + " · reported at " + time(event.available_samples) : start + " to " + time(event.end);
    });
    get("event-list").replaceChildren(...(descriptions.length ? descriptions : ["No wake events recorded."]).map(text => {
      const row = document.createElement("li");
      row.textContent = text;
      return row;
    }));
    get("test-output").hidden = false;
  }

  audio.addEventListener("timeupdate", () => {
    if (seconds) cursor.style.left = Math.min(100, 100 * audio.currentTime / seconds) + "%";
  });

  async function load() {
    try {
      const response = await fetch("./examples.json");
      if (!response.ok) throw new Error("Examples unavailable");
      const data = await response.json();
      for (const [index, example] of data.examples.entries()) workflow.add(new Option(example.title, String(index)));
      const show = () => {
        const example = data.examples[Number(workflow.value)];
        render(example, example.conditions.find(clip => clip.condition === condition.value), data.sample_rate);
      };
      workflow.addEventListener("change", show);
      condition.addEventListener("change", show);
      show();
      workflow.disabled = false;
      condition.disabled = false;
    } catch {
      status.textContent = "Replay could not load. Open the recorded event data to see the results.";
    }
  }

  load();
})();
