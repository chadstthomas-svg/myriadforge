(() => {
  const hardwareSections = ["device", "ceiling", "architecture", "apps", "quickstart"];
  if (hardwareSections.includes(location.hash.slice(1))) {
    location.replace(new URL("./hardware/index.html" + location.hash, location.href));
    return;
  }
  const demo = document.querySelector(".launch iframe");
  const origin = new URL(demo.src).origin;
  window.addEventListener("message", (event) => {
    if (event.origin !== origin || event.source !== demo.contentWindow ||
        event.data?.type !== "racklab:bankshot-height" || !Number.isFinite(event.data.height)) return;
    demo.style.height = Math.min(1800, Math.max(350, event.data.height)) + "px";
  });
})();
