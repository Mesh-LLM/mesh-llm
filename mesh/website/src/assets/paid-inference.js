(function () {
  "use strict";

  var ROOT_SELECTOR = "[data-paid-inference]";
  var TAB_SELECTOR = "[data-payment-tab]";
  var PANEL_SELECTOR = "[data-payment-panel]";
  var MAP_SELECTOR = "[data-payment-map]";
  var REDUCE_QUERY = "(prefers-reduced-motion: reduce)";
  var MOBILE_QUERY = "(max-width: 720px)";
  var ENTER_THRESHOLD = 0.1;
  var ENTER_ROOT_MARGIN = "0px 0px -15% 0px";
  var PHASES = [
    { route: "request", at: 140, duration: 540 },
    { route: "deposit", at: 820, duration: 540 },
    { route: "stream", at: 1520, duration: 760 },
    { route: "settlement", at: 2440, duration: 620 },
  ];
  var COMPLETE_AT = 3220;

  var reduceMotion = window.matchMedia(REDUCE_QUERY);
  var mobileLayout = window.matchMedia(MOBILE_QUERY);

  function toArray(list) {
    return Array.prototype.slice.call(list || []);
  }

  function getAnime() {
    return window.anime && window.anime.createTimeline ? window.anime : null;
  }

  function getMode(root) {
    return root.getAttribute("data-payment-mode") || "earn";
  }

  function getActiveMap(root, mode) {
    return root.querySelector('[data-payment-map="' + mode + '"]');
  }

  function getActivePanel(root, mode) {
    return root.querySelector('[data-payment-panel="' + mode + '"]');
  }

  function setVisible(collection, mode, attribute) {
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

  function setStatus(root, text) {
    var status = root.querySelector("[data-payment-status]");
    if (status) status.textContent = text;
  }

  function setMotionControl(root, action) {
    var control = root.querySelector("[data-payment-motion]");
    if (!control) return;

    var labels = {
      pause: "Pause animation",
      play: "Play animation",
      replay: "Replay animation",
      resume: "Resume animation",
    };
    var visible = Boolean(action) && !reduceMotion.matches;
    control.hidden = !visible;
    control.setAttribute("aria-pressed", String(action === "resume"));
    control.setAttribute("data-payment-motion-action", action || "");

    var label = control.querySelector("[data-payment-motion-label]");
    if (label && labels[action]) label.textContent = labels[action];
  }

  function resumeTimeline(timeline) {
    if (typeof timeline.resume === "function") {
      timeline.resume();
    } else {
      timeline.play();
    }
  }

  function directionDistance(packet) {
    var track = packet.closest(".paid-inference-track");
    if (!track) return 0;

    if (mobileLayout.matches) {
      return Math.max(0, track.clientHeight - packet.offsetHeight);
    }

    return Math.max(0, track.clientWidth - packet.offsetWidth);
  }

  function packetCoordinates(packet) {
    var distance = directionDistance(packet);
    var isReverse = packet.getAttribute("data-payment-direction") === "reverse";
    return {
      property: mobileLayout.matches ? "translateY" : "translateX",
      from: isReverse ? distance : 0,
      to: isReverse ? 0 : distance,
    };
  }

  function positionPacket(packet, atDestination) {
    var coordinates = packetCoordinates(packet);
    var value = atDestination ? coordinates.to : coordinates.from;
    packet.style.opacity = "0";
    packet.style.transform = coordinates.property === "translateY"
      ? "translateY(" + value + "px)"
      : "translateX(" + value + "px)";
  }

  function setSteps(panel, state) {
    if (!panel) return;

    toArray(panel.querySelectorAll("[data-payment-step]")).forEach(function (step) {
      var name = step.getAttribute("data-payment-step");
      var isActive = false;
      var isComplete = false;

      if (state === "complete") {
        isComplete = true;
      } else if (state === "request") {
        isComplete = name === "pricing";
        isActive = name === "deposit";
      } else if (state === "deposit") {
        isComplete = name === "pricing";
        isActive = name === "deposit";
      } else if (state === "stream" || state === "settlement") {
        isComplete = name === "pricing" || name === "deposit";
        isActive = name === "settle";
      }

      step.classList.toggle("is-active", isActive);
      step.classList.toggle("is-complete", isComplete);
    });
  }

  function clearVisual(root, mode) {
    var map = getActiveMap(root, mode);
    var panel = getActivePanel(root, mode);
    if (!map) return;

    toArray(map.querySelectorAll("[data-payment-route]")).forEach(function (route) {
      route.classList.remove("is-active", "is-complete");
    });
    toArray(map.querySelectorAll("[data-payment-packet]")).forEach(function (packet) {
      positionPacket(packet, false);
    });
    setSteps(panel, "idle");
    root.setAttribute("data-payment-state", "idle");
    setStatus(root, mode + " flow ready");
  }

  function completeVisual(root, mode) {
    var map = getActiveMap(root, mode);
    var panel = getActivePanel(root, mode);
    if (!map) return;

    toArray(map.querySelectorAll("[data-payment-route]")).forEach(function (route) {
      route.classList.remove("is-active");
      route.classList.add("is-complete");
    });
    toArray(map.querySelectorAll("[data-payment-packet]")).forEach(function (packet) {
      positionPacket(packet, true);
    });
    setSteps(panel, "complete");
    root.setAttribute("data-payment-state", "complete");
    setStatus(root, mode + " flow complete");
  }

  function markPhase(root, mode, phase) {
    var map = getActiveMap(root, mode);
    var panel = getActivePanel(root, mode);
    if (!map) return;

    var phaseIndex = PHASES.findIndex(function (item) {
      return item.route === phase;
    });

    toArray(map.querySelectorAll("[data-payment-route]")).forEach(function (route) {
      var routeName = route.getAttribute("data-payment-route");
      var routeIndex = PHASES.findIndex(function (item) {
        return item.route === routeName;
      });
      route.classList.toggle("is-active", routeName === phase);
      route.classList.toggle("is-complete", routeIndex > -1 && routeIndex < phaseIndex);
    });

    setSteps(panel, phase);
    root.setAttribute("data-payment-state", phase);

    var labels = {
      request: "request routed over Mesh QUIC",
      deposit: "first-request deposit over Lightning",
      stream: "response streaming over Mesh QUIC",
      settlement: "remaining balance settled over Lightning",
    };
    setStatus(root, mode + " · " + labels[phase]);
  }

  function stopRun(root) {
    root.__paidInferenceRun = (root.__paidInferenceRun || 0) + 1;

    if (root.__paidInferenceTimeline && typeof root.__paidInferenceTimeline.pause === "function") {
      root.__paidInferenceTimeline.pause();
    }
    root.__paidInferenceTimeline = null;
    root.__paidInferenceAutosequencing = false;
    root.__paidInferencePausedByUser = false;
    root.__paidInferencePausedByViewport = false;
    setMotionControl(root, null);
    return root.__paidInferenceRun;
  }

  function selectMode(root, mode) {
    var tabs = toArray(root.querySelectorAll(TAB_SELECTOR));
    var panels = toArray(root.querySelectorAll(PANEL_SELECTOR));
    var maps = toArray(root.querySelectorAll(MAP_SELECTOR));
    var validModes = tabs.map(function (tab) {
      return tab.getAttribute("data-payment-tab");
    });

    if (validModes.indexOf(mode) === -1) mode = validModes[0] || "earn";

    tabs.forEach(function (tab) {
      var isActive = tab.getAttribute("data-payment-tab") === mode;
      tab.classList.toggle("is-active", isActive);
      tab.setAttribute("aria-selected", String(isActive));
      tab.setAttribute("tabindex", isActive ? "0" : "-1");
    });

    setVisible(panels, mode, "data-payment-panel");
    setVisible(maps, mode, "data-payment-map");
    root.setAttribute("data-payment-mode", mode);
    clearVisual(root, mode);
    return mode;
  }

  function animatePacket(timeline, packet, phase) {
    var coordinates = packetCoordinates(packet);
    var transform = {};
    transform[coordinates.property] = [coordinates.from, coordinates.to];

    timeline.add(packet, {
      opacity: [0, 1],
      duration: 120,
      ease: "linear",
      composition: "replace",
    }, phase.at);

    transform.duration = phase.duration;
    transform.ease = "inOutQuad";
    transform.composition = "replace";
    timeline.add(packet, transform, phase.at);

    timeline.add(packet, {
      opacity: [1, 0],
      duration: 150,
      ease: "outQuad",
      composition: "replace",
    }, phase.at + phase.duration - 120);
  }

  function playMode(root, mode, token, onComplete) {
    var animeApi = getAnime();
    var map = getActiveMap(root, mode);

    if (!animeApi || reduceMotion.matches || !map) {
      completeVisual(root, mode);
      if (onComplete) onComplete();
      return;
    }

    clearVisual(root, mode);
    root.setAttribute("data-payment-state", "running");

    var timeline = animeApi.createTimeline({
      autoplay: false,
      defaults: { ease: "linear" },
    });
    root.__paidInferenceTimeline = timeline;
    root.__paidInferencePausedByUser = false;
    root.__paidInferencePausedByViewport = false;
    setMotionControl(root, "pause");

    PHASES.forEach(function (phase) {
      var route = map.querySelector('[data-payment-route="' + phase.route + '"]');
      var packet = route && route.querySelector("[data-payment-packet]");

      timeline.call(function () {
        if (root.__paidInferenceRun !== token) return;
        markPhase(root, mode, phase.route);
      }, phase.at);

      if (packet) animatePacket(timeline, packet, phase);
    });

    timeline.call(function () {
      if (root.__paidInferenceRun !== token) return;
      completeVisual(root, mode);
      root.__paidInferenceTimeline = null;
      root.__paidInferencePausedByUser = false;
      root.__paidInferencePausedByViewport = false;
      setMotionControl(root, "replay");
      if (onComplete) onComplete();
    }, COMPLETE_AT);

    timeline.play();
  }

  function replayMode(root, mode) {
    var token = stopRun(root);
    mode = selectMode(root, mode);
    root.setAttribute("data-payment-autoplayed", "true");
    playMode(root, mode, token);
  }

  function playAutosequence(root) {
    var token = stopRun(root);
    root.__paidInferenceAutosequencing = true;
    root.setAttribute("data-payment-autoplayed", "true");
    var mode = selectMode(root, "earn");

    playMode(root, mode, token, function () {
      if (root.__paidInferenceRun !== token) return;

      mode = selectMode(root, "spend");
      playMode(root, mode, token, function () {
        if (root.__paidInferenceRun !== token) return;
        root.__paidInferenceAutosequencing = false;
        if (root.__paidInferenceObserver) {
          root.__paidInferenceObserver.disconnect();
          root.__paidInferenceObserver = null;
        }
      });
    });
  }

  function bindTabs(root, tabs) {
    tabs.forEach(function (tab, index) {
      tab.addEventListener("focus", function () {
        if (!root.__paidInferenceAutosequencing) return;

        stopRun(root);
        var mode = selectMode(root, tab.getAttribute("data-payment-tab"));
        completeVisual(root, mode);
        setMotionControl(root, "replay");
      });

      tab.addEventListener("click", function () {
        replayMode(root, tab.getAttribute("data-payment-tab"));
      });

      tab.addEventListener("keydown", function (event) {
        var nextIndex = index;
        var handled = false;
        if (event.key === "ArrowRight" || event.key === "ArrowDown") {
          nextIndex = (index + 1) % tabs.length;
          handled = true;
        }
        if (event.key === "ArrowLeft" || event.key === "ArrowUp") {
          nextIndex = (index - 1 + tabs.length) % tabs.length;
          handled = true;
        }
        if (event.key === "Home") {
          nextIndex = 0;
          handled = true;
        }
        if (event.key === "End") {
          nextIndex = tabs.length - 1;
          handled = true;
        }
        if (!handled) return;

        event.preventDefault();
        if (nextIndex === index) return;

        tabs[nextIndex].focus();
        replayMode(root, tabs[nextIndex].getAttribute("data-payment-tab"));
      });
    });
  }

  function bindMotionControl(root) {
    var control = root.querySelector("[data-payment-motion]");
    if (!control) return;

    control.addEventListener("click", function () {
      var timeline = root.__paidInferenceTimeline;
      if (!timeline) {
        replayMode(root, getMode(root));
        control.focus();
        return;
      }

      if (root.__paidInferencePausedByUser) {
        root.__paidInferencePausedByUser = false;
        if (!root.__paidInferencePausedByViewport) {
          resumeTimeline(timeline);
          root.setAttribute("data-payment-state", "running");
        }
      } else {
        root.__paidInferencePausedByUser = true;
        timeline.pause();
        root.setAttribute("data-payment-state", "paused");
      }

      setMotionControl(root, root.__paidInferencePausedByUser || root.__paidInferencePausedByViewport ? "resume" : "pause");
    });
  }

  function bindViewport(root) {
    if (reduceMotion.matches || !getAnime() || !("IntersectionObserver" in window)) {
      completeVisual(root, getMode(root));
      setMotionControl(root, null);
      return;
    }

    setMotionControl(root, "play");

    root.__paidInferenceObserver = new IntersectionObserver(function (entries) {
      entries.forEach(function (entry) {
        root.__paidInferenceVisible = entry.isIntersecting;

        if (entry.isIntersecting && !root.hasAttribute("data-payment-autoplayed")) {
          if (root.contains(document.activeElement) && document.activeElement.matches(TAB_SELECTOR)) {
            root.setAttribute("data-payment-autoplayed", "true");
            completeVisual(root, getMode(root));
            setMotionControl(root, "replay");
            root.__paidInferenceObserver.disconnect();
            root.__paidInferenceObserver = null;
            return;
          }
          playAutosequence(root);
          return;
        }

        if (!root.__paidInferenceTimeline) return;

        if (!entry.isIntersecting) {
          root.__paidInferencePausedByViewport = true;
          root.__paidInferenceTimeline.pause();
          root.setAttribute("data-payment-state", "paused");
          setMotionControl(root, "resume");
        } else if (root.__paidInferencePausedByViewport) {
          root.__paidInferencePausedByViewport = false;
          if (!root.__paidInferencePausedByUser) {
            resumeTimeline(root.__paidInferenceTimeline);
            root.setAttribute("data-payment-state", "running");
          }
          setMotionControl(root, root.__paidInferencePausedByUser ? "resume" : "pause");
        }
      });
    }, {
      rootMargin: ENTER_ROOT_MARGIN,
      threshold: ENTER_THRESHOLD,
    });

    root.__paidInferenceObserver.observe(root);
  }

  function bindPreferenceChanges(root) {
    if (reduceMotion.addEventListener) {
      reduceMotion.addEventListener("change", function (event) {
        if (!event.matches) return;
        var control = root.querySelector("[data-payment-motion]");
        if (control && document.activeElement === control) {
          var activeTab = root.querySelector(TAB_SELECTOR + '[aria-selected="true"]');
          if (activeTab) activeTab.focus();
        }
        stopRun(root);
        completeVisual(root, getMode(root));
      });
    }

    if (mobileLayout.addEventListener) {
      mobileLayout.addEventListener("change", function () {
        var control = root.querySelector("[data-payment-motion]");
        var restoreControlFocus = control && document.activeElement === control;
        stopRun(root);
        completeVisual(root, getMode(root));
        if (!reduceMotion.matches && getAnime()) setMotionControl(root, "replay");
        if (restoreControlFocus) control.focus();
      });
    }
  }

  function init(root) {
    var tabs = toArray(root.querySelectorAll(TAB_SELECTOR));
    if (!tabs.length) return;

    bindTabs(root, tabs);
    bindMotionControl(root);
    selectMode(root, getMode(root));
    bindPreferenceChanges(root);
    bindViewport(root);
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
