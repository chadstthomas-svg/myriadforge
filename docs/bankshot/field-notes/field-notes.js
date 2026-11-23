(() => {
  const data = window.bankshotReplay;
  const el = (id) => document.getElementById(id);
  const tabs = el("case-selector");
  const modes = Array.from(document.querySelectorAll("[data-runtime]"));
  let current = data.cases[0];
  let runtime = "ncs1";
  const results = Object.fromEntries(Object.entries(data.backends).map(([name, report]) =>
    [name, Object.fromEntries(report.results.map((row) => [row.id, row]))]));
  const link = (node, href) => { node.href = href; };
  // These marks are keyed to the original DVIDS video IDs.
  const watchCues = {
    727461: {
      at: 118, from: 116, until: 120.6,
      box: [49, 41, 22, 45],
      label: "Lockout hardware",
      detail: "Watch for the red lockout hardware on the machine disconnect during the training demonstration.",
    },
    702029: {
      at: 27, from: 24, until: 38.5,
      box: [33, 7, 55, 76],
      label: "Rotating chuck",
      detail: "Look at the spinning chuck on the right. The separate OSHA citation concerns guarding this type of rotating part.",
    },
    851044: {
      at: 1178, from: 1170, until: 1190,
      box: [21, 30, 57, 55],
      label: "Storefront doorway",
      detail: "The box frames the public entrance and nearby rack at Quantico. The cited blockage was at a different store’s stockroom exit.",
    },
  };

  function timestamp(seconds) {
    return Math.floor(seconds / 60) + ":" + String(seconds % 60).padStart(2, "0");
  }

  for (const one of data.cases) {
    const tab = document.createElement("a");
    tab.href = "#" + one.id;
    tab.textContent = one.tab;
    tab.dataset.case = one.id;
    tabs.append(tab);
  }

  function setupMedia(media) {
    const container = el("field-media");
    container.querySelector("video")?.pause();
    const cue = watchCues[media.id];
    let player;
    const poster = document.createElement("img");
    poster.alt = media.title;
    poster.addEventListener("error", () => {
      if (!poster.isConnected) return;
      poster.hidden = true;
      const name = document.createElement("span");
      name.textContent = media.title;
      container.prepend(name);
    }, {once: true});
    container.replaceChildren(poster);
    poster.src = media.poster;

    function openVideo() {
      if (player?.isConnected) return player;
      const video = document.createElement("video");
      video.controls = true;
      video.playsInline = true;
      video.poster = media.poster;
      video.setAttribute("aria-label", media.title);
      video.addEventListener("error", () => {
        if (!video.isConnected) return;
        container.replaceChildren(poster, play);
        el("media-status").textContent = "Video unavailable here. Open the original on DVIDS.";
      }, {once: true});
      el("media-status").textContent = "";
      container.replaceChildren(video);
      if (cue) {
        const marker = document.createElement("div");
        marker.className = "watch-marker";
        marker.setAttribute("aria-hidden", "true");
        const label = document.createElement("div");
        label.textContent = cue.label;
        marker.append(label);
        const [left, top, width, height] = cue.box;
        Object.assign(marker.style, {left: left + "%", top: top + "%", width: width + "%", height: height + "%"});
        const updateMarker = () => {
          marker.hidden = video.seeking || video.readyState < 2 ||
            video.currentTime < cue.from || video.currentTime >= cue.until;
        };
        for (const event of ["loadeddata", "seeking", "seeked", "timeupdate"]) {
          video.addEventListener(event, updateMarker);
        }
        updateMarker();
        container.append(marker);
      }
      video.src = media.stream;
      player = video;
      return video;
    }

    const play = document.createElement("button");
    if (media.kind === "video") {
      play.type = "button";
      play.textContent = "Play original footage";
      play.addEventListener("click", () => {
        const video = openVideo();
        video.focus();
        video.play().catch(() => {});
      });
      container.append(play);
    }

    const jump = el("watch-jump");
    el("watch-moment").hidden = !cue;
    el("watch-copy").textContent = cue?.detail ?? "";
    jump.textContent = cue ? "Jump to " + timestamp(cue.at) : "";
    jump.onclick = cue ? () => {
      const video = openVideo();
      video.pause();
      video.currentTime = cue.at;
      container.scrollIntoView({block: "nearest"});
    } : null;
    el("media-status").textContent = "";
    el("media-context").textContent = media.label;
    el("media-credit").textContent = media.credit;
    link(el("media-source"), media.url);
  }

  function renderRun() {
    const row = results[runtime][current.id];
    const gate = row.target_rank === null ? "Missed · Outside top 10" : "Found · Rank " + row.target_rank;
    const badge = el("target-rank");
    badge.dataset.miss = String(row.target_rank === null);
    const text = document.createElement("strong");
    const reference = document.createElement("span");
    text.textContent = gate;
    reference.textContent = current.reference;
    badge.replaceChildren(text, reference);
    const list = el("hits");
    list.replaceChildren(...row.hits.slice(0, 3).map((item) => {
      const li = document.createElement("li");
      const a = document.createElement("a");
      a.textContent = item.title;
      a.href = item.url;
      a.target = "_blank";
      a.rel = "noopener";
      li.append(a);
      return li;
    }));
    el("run-meta").textContent = row.median_ms.toFixed(2) + " ms median across " + row.requests + " warm requests";
    modes.forEach((button) => button.setAttribute("aria-pressed", String(button.dataset.runtime === runtime)));
  }

  function renderCase() {
    const id = window.location.hash.slice(1);
    if (id === "case-view" && tabs.querySelector("[aria-current]")) return;
    current = data.cases.find((item) => item.id === id) || data.cases[0];
    for (const tab of tabs.children) {
      if (tab.dataset.case === current.id) tab.setAttribute("aria-current", "page");
      else tab.removeAttribute("aria-current");
    }
    el("case-sector").textContent = current.sector;
    el("case-title").textContent = current.title;
    el("case-query").textContent = current.note;
    const vehicle = el("vehicle-note");
    vehicle.hidden = !current.vehicle;
    vehicle.textContent = current.vehicle ? "Vehicle fields · " + current.vehicle.join(" ") : "";
    el("record-type").textContent = current.record.label;
    el("record-text").textContent = current.record.summary;
    link(el("record-link"), current.record.url);
    setupMedia(current.media);
    renderRun();
  }

  modes.forEach((mode) => mode.addEventListener("click", () => {
    runtime = mode.dataset.runtime;
    renderRun();
  }));
  window.addEventListener("hashchange", renderCase);
  renderCase();
})();
