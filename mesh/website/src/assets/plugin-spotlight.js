(function () {
  var SPOTLIGHT_SELECTOR = "[data-plugin-spotlight]";
  var REDUCE_QUERY = "(prefers-reduced-motion: reduce)";
  var INTERVAL = 6000;

  function toArray(list) {
    return Array.prototype.slice.call(list || []);
  }

  function getAnime() {
    return window.anime && window.anime.animate ? window.anime : null;
  }

  function setup(root) {
    var tablist = root.querySelector("[data-plugin-spot-tabs]");
    var tabs = toArray(root.querySelectorAll("[data-plugin-spot-tab]"));
    var panels = toArray(root.querySelectorAll("[data-plugin-spot-panel]"));
    if (!tablist || tabs.length === 0 || tabs.length !== panels.length) return;

    var animeApi = getAnime();
    var reduceMotion = window.matchMedia(REDUCE_QUERY);
    var current = 0;
    var autoplay = !reduceMotion.matches;
    var hovered = false;
    var focused = false;
    var visible = false;
    var progress = null;

    tablist.hidden = false;
    root.classList.add("is-interactive");

    function stopProgress() {
      if (progress) progress.pause();
      progress = null;
      tabs.forEach(function (tab) {
        var bar = tab.querySelector("[data-plugin-spot-progress]");
        if (bar) bar.style.width = "0";
      });
    }

    function runProgress() {
      stopProgress();
      if (!autoplay || !animeApi) return;
      var bar = tabs[current].querySelector("[data-plugin-spot-progress]");
      if (!bar) return;
      progress = animeApi.animate(bar, {
        width: ["0%", "100%"],
        duration: INTERVAL,
        ease: "linear",
        onComplete: function () { select((current + 1) % tabs.length, false); },
      });
      syncPause();
    }

    function syncPause() {
      if (!progress) return;
      if (hovered || focused || !visible || document.hidden) progress.pause();
      else progress.play();
    }

    function select(index, fromUser) {
      if (fromUser) autoplay = false;
      current = index;
      tabs.forEach(function (tab, i) {
        var active = i === index;
        tab.setAttribute("aria-selected", active ? "true" : "false");
        tab.tabIndex = active ? 0 : -1;
        panels[i].hidden = !active;
      });
      if (animeApi && !reduceMotion.matches) {
        animeApi.animate(panels[index], { opacity: [0, 1], translateY: [8, 0], duration: 320, ease: "outQuad" });
      }
      if (!fromUser) {
        // Keep the active tab in view on narrow screens without moving the page.
        var tab = tabs[index];
        tablist.scrollTo({ left: tab.offsetLeft - tablist.offsetLeft, behavior: reduceMotion.matches ? "auto" : "smooth" });
      }
      runProgress();
    }

    tabs.forEach(function (tab, i) {
      tab.addEventListener("click", function () { select(i, true); });
      tab.addEventListener("keydown", function (event) {
        var next = null;
        if (event.key === "ArrowDown" || event.key === "ArrowRight") next = (i + 1) % tabs.length;
        if (event.key === "ArrowUp" || event.key === "ArrowLeft") next = (i - 1 + tabs.length) % tabs.length;
        if (event.key === "Home") next = 0;
        if (event.key === "End") next = tabs.length - 1;
        if (next === null) return;
        event.preventDefault();
        select(next, true);
        tabs[next].focus();
      });
    });

    // Pause while someone is reading the panel, not while the cursor merely crosses the tabs.
    var panelBox = root.querySelector(".plugin-spot-panels");
    panelBox.addEventListener("mouseenter", function () { hovered = true; syncPause(); });
    panelBox.addEventListener("mouseleave", function () { hovered = false; syncPause(); });
    panelBox.addEventListener("focusin", function () { focused = true; syncPause(); });
    panelBox.addEventListener("focusout", function () { focused = false; syncPause(); });
    document.addEventListener("visibilitychange", syncPause);

    if ("IntersectionObserver" in window) {
      new IntersectionObserver(function (entries) {
        visible = entries[0].isIntersecting;
        syncPause();
      }, { threshold: 0.25 }).observe(root);
    } else {
      visible = true;
    }

    reduceMotion.addEventListener("change", function (event) {
      if (event.matches) {
        autoplay = false;
        stopProgress();
      }
    });

    // Deep links such as /plugins/#plugin-panel-uzu open that plugin and stop autoplay.
    var linked = panels.findIndex(function (panel) { return "#" + panel.id === window.location.hash; });
    if (linked >= 0) {
      select(linked, true);
      root.scrollIntoView({ block: "start" });
    } else {
      select(0, false);
    }
  }

  function init() {
    toArray(document.querySelectorAll(SPOTLIGHT_SELECTOR)).forEach(setup);
  }

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init);
  else init();
})();
