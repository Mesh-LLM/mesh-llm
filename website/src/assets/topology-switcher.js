(function () {
  "use strict";

  var ROOT_SELECTOR = "[data-topology-switcher]";
  var TAB_SELECTOR = "[data-topology-mode]";
  var PANEL_SELECTOR = "[data-topology-panel]";
  var MAP_SELECTOR = "[data-topology-map]";
  var REDUCE_QUERY = "(prefers-reduced-motion: reduce)";

  var reduceMotion = window.matchMedia(REDUCE_QUERY);

  function toArray(list) {
    return Array.prototype.slice.call(list || []);
  }

  function getAnime() {
    return window.anime && window.anime.animate ? window.anime : null;
  }

  function getStagger(animeApi) {
    if (!animeApi) return null;
    return animeApi.stagger || (animeApi.utils && animeApi.utils.stagger) || null;
  }

  function setActive(collection, mode, attribute) {
    collection.forEach(function (item) {
      var isActive = item.getAttribute(attribute) === mode;
      item.classList.toggle("is-active", isActive);
      if (isActive) {
        item.removeAttribute("hidden");
      } else {
        item.setAttribute("hidden", "");
      }
    });
  }

  function animateMode(root, mode) {
    if (reduceMotion.matches) return;

    var animeApi = getAnime();
    if (!animeApi) return;

    var activePanel = root.querySelector('[data-topology-panel="' + mode + '"]');
    var activeMap = root.querySelector('[data-topology-map="' + mode + '"]');
    if (!activePanel || !activeMap) return;

    var nodes = activeMap ? activeMap.querySelectorAll(".mesh-node, .mesh-core, .split-stage, .split-callout") : [];
    var stagger = getStagger(animeApi);

    animeApi.animate(activePanel, {
      opacity: [0, 1],
      translateY: [10, 0],
      duration: 420,
      ease: "outQuart"
    });

    animeApi.animate(activeMap, {
      opacity: [0, 1],
      translateY: [12, 0],
      duration: 460,
      ease: "outQuart"
    });

    // These nodes use CSS transforms for absolute centering. Animating translateY
    // here would overwrite that centering and pull the route lines out of alignment.
    animeApi.animate(nodes, {
      opacity: [0, 1],
      duration: 430,
      delay: stagger ? stagger(48) : 0,
      ease: "outQuart"
    });
  }

  function setMode(root, mode) {
    var tabs = toArray(root.querySelectorAll(TAB_SELECTOR));
    var panels = toArray(root.querySelectorAll(PANEL_SELECTOR));
    var maps = toArray(root.querySelectorAll(MAP_SELECTOR));
    var validModes = tabs
      .map(function (tab) { return tab.getAttribute("data-topology-mode"); })
      .filter(Boolean);
    if (validModes.indexOf(mode) === -1) {
      mode = validModes.indexOf("route") !== -1 ? "route" : validModes[0];
    }

    tabs.forEach(function (tab) {
      var isActive = tab.getAttribute("data-topology-mode") === mode;
      tab.classList.toggle("is-active", isActive);
      tab.setAttribute("aria-selected", String(isActive));
      tab.setAttribute("tabindex", isActive ? "0" : "-1");
    });

    setActive(panels, mode, "data-topology-panel");
    setActive(maps, mode, "data-topology-map");
    root.setAttribute("data-topology-active", mode);
    animateMode(root, mode);
  }

  function bindKeyboard(root, tabs) {
    tabs.forEach(function (tab, index) {
      tab.addEventListener("keydown", function (event) {
        var nextIndex = index;
        if (event.key === "ArrowRight" || event.key === "ArrowDown") nextIndex = (index + 1) % tabs.length;
        if (event.key === "ArrowLeft" || event.key === "ArrowUp") nextIndex = (index - 1 + tabs.length) % tabs.length;
        if (event.key === "Home") nextIndex = 0;
        if (event.key === "End") nextIndex = tabs.length - 1;
        if (nextIndex === index) return;

        event.preventDefault();
        tabs[nextIndex].focus();
        setMode(root, tabs[nextIndex].getAttribute("data-topology-mode"));
      });
    });
  }

  function init(root) {
    var tabs = toArray(root.querySelectorAll(TAB_SELECTOR));
    if (!tabs.length) return;

    tabs.forEach(function (tab) {
      tab.addEventListener("click", function () {
        setMode(root, tab.getAttribute("data-topology-mode"));
      });
    });

    bindKeyboard(root, tabs);
    setMode(root, root.getAttribute("data-topology-active") || "route");
  }

  function boot() {
    toArray(document.querySelectorAll(ROOT_SELECTOR)).forEach(init);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", boot);
  } else {
    boot();
  }
})();
