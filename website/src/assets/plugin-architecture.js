(function () {
  var DIAGRAM_SELECTOR = "[data-plugin-architecture]";
  var DESKTOP_QUERY = "(min-width: 901px)";
  var REDUCE_QUERY = "(prefers-reduced-motion: reduce)";
  var SNAP_DURATION = 560;
  var SOCKET_HIGHLIGHT_DURATION = 260;
  var PAIR_REST = 80;
  var PAIR_INTERVAL = SNAP_DURATION + SOCKET_HIGHLIGHT_DURATION + PAIR_REST;
  var IN_VIEW_DELAY = 650;
  var DRAW_DELAY = 520;
  var DRAW_DURATION = 820;

  var desktopMedia = window.matchMedia(DESKTOP_QUERY);
  var reduceMotion = window.matchMedia(REDUCE_QUERY);
  var resizeTimer = null;

  function toArray(list) {
    return Array.prototype.slice.call(list || []);
  }

  function getAnime() {
    return window.anime && window.anime.animate ? window.anime : null;
  }

  function getCreateTimeline(animeApi) {
    return animeApi && animeApi.createTimeline ? animeApi.createTimeline : null;
  }

  function getCreateDrawable(animeApi) {
    if (animeApi.svg && animeApi.svg.createDrawable) {
      return animeApi.svg.createDrawable;
    }

    return animeApi.createDrawable || null;
  }

  function getActiveSvg(diagram) {
    return diagram.querySelector(desktopMedia.matches ? ".arch-svg--full" : ".arch-svg--small");
  }

  function getAllSvgs(diagram) {
    return toArray(diagram.querySelectorAll(".arch-svg"));
  }

  function getPlugSeat(plug) {
    var seat = Number.parseFloat(plug.getAttribute("data-arch-seat"));
    return Number.isFinite(seat) ? seat : 0;
  }

  function getPlugOvershoot(plug) {
    var overshoot = Number.parseFloat(plug.getAttribute("data-arch-overshoot"));
    return Number.isFinite(overshoot) ? overshoot : 0;
  }

  function setPlugY(plug, y) {
    plug.setAttribute("transform", "translate(0 " + y.toFixed(2) + ")");
  }

  function clearSvg(svg) {
    toArray(svg.querySelectorAll("[data-arch-plug]")).forEach(function (plug) {
      plug.classList.remove("is-clicking");
      setPlugY(plug, 0);
    });

    toArray(svg.querySelectorAll(".arch-svg-socket-glow, .arch-svg-runtime-glow, .arch-svg-draw-path")).forEach(function (node) {
      node.style.opacity = "";
      node.removeAttribute("stroke-dasharray");
      node.removeAttribute("stroke-dashoffset");
      node.removeAttribute("pathLength");
    });
  }

  function clearDiagram(diagram) {
    getAllSvgs(diagram).forEach(clearSvg);
    diagram.classList.remove("is-clicked");
    delete diagram.dataset.pluginArchitectureState;
  }

  function setSeated(diagram) {
    getAllSvgs(diagram).forEach(function (svg) {
      toArray(svg.querySelectorAll("[data-arch-plug]")).forEach(function (plug) {
        plug.classList.remove("is-clicking");
        setPlugY(plug, getPlugSeat(plug));
      });
      toArray(svg.querySelectorAll(".arch-svg-socket-glow, .arch-svg-runtime-glow, .arch-svg-draw-path")).forEach(function (node) {
        node.style.opacity = "";
      });
    });

    diagram.classList.add("is-clicked");
    diagram.dataset.pluginArchitectureState = "complete";
  }

  function getPairs(svg) {
    var map = {};

    toArray(svg.querySelectorAll("[data-arch-plug]")).forEach(function (plug) {
      var pair = plug.getAttribute("data-arch-pair") || "0";
      if (!map[pair]) map[pair] = [];
      map[pair].push({
        plug: plug,
        seat: getPlugSeat(plug),
        overshoot: getPlugOvershoot(plug),
        socket: svg.querySelector('[data-arch-socket-glow="' + plug.getAttribute("data-arch-socket") + '"]'),
      });
    });

    return Object.keys(map).sort(function (a, b) {
      return Number(a) - Number(b);
    }).map(function (key) {
      return map[key];
    });
  }

  function lerp(from, to, progress) {
    return from + (to - from) * progress;
  }

  function easeOut(power, progress) {
    return 1 - Math.pow(1 - progress, power);
  }

  function plugYAt(progress, seat, overshoot) {
    var sign = seat < 0 ? -1 : 1;
    var pushPoint = seat - sign * overshoot;

    if (progress < 0.72) {
      return lerp(0, pushPoint, easeOut(3, progress / 0.72));
    }

    if (progress < 0.84) {
      return pushPoint;
    }

    return lerp(pushPoint, seat, easeOut(6, (progress - 0.84) / 0.16));
  }

  function drawableTargets(svg, animeApi) {
    var createDrawable = getCreateDrawable(animeApi);
    var paths = toArray(svg.querySelectorAll(".arch-svg-draw-path"));

    if (!createDrawable || !paths.length) {
      return [];
    }

    return paths.reduce(function (targets, path) {
      return targets.concat(createDrawable(path));
    }, []);
  }

  function buildTimeline(diagram, svg, animeApi) {
    var createTimeline = getCreateTimeline(animeApi);
    var pairs = getPairs(svg);
    var drawPaths = toArray(svg.querySelectorAll(".arch-svg-draw-path"));
    var drawables;
    var timeline;
    var drawAt;

    if (!createTimeline || !pairs.length) {
      return null;
    }

    clearSvg(svg);
    drawables = drawableTargets(svg, animeApi);
    timeline = createTimeline({
      autoplay: false,
      defaults: { ease: "linear" },
    });

    pairs.forEach(function (pair, pairIndex) {
      var pairStart = pairIndex * PAIR_INTERVAL;

      pair.forEach(function (item) {
        var state = { progress: 0 };

        timeline.add(state, {
          progress: [0, 1],
          duration: SNAP_DURATION,
          ease: "linear",
          composition: "replace",
          onBegin: function () {
            item.plug.classList.add("is-clicking");
          },
          onUpdate: function () {
            setPlugY(item.plug, plugYAt(state.progress, item.seat, item.overshoot));
          },
          onComplete: function () {
            setPlugY(item.plug, item.seat);
            item.plug.classList.remove("is-clicking");
          },
        }, pairStart);

        if (item.socket) {
          timeline.add(item.socket, {
            opacity: [0, 1, 0],
            duration: SOCKET_HIGHLIGHT_DURATION,
            ease: "out(3)",
            composition: "replace",
          }, pairStart + SNAP_DURATION);
        }
      });

    });

    drawAt = pairs.length * PAIR_INTERVAL + DRAW_DELAY;

    timeline.call(function () {
      diagram.classList.add("is-clicked");
      diagram.dataset.pluginArchitectureState = "drawing";
    }, pairs.length * PAIR_INTERVAL);

    if (drawables.length) {
      timeline.add(drawPaths, {
        opacity: [0, 1],
        duration: 80,
        ease: "linear",
        composition: "replace",
      }, drawAt);
      timeline.add(drawables, {
        draw: ["0 0", "0 1"],
        duration: DRAW_DURATION,
        ease: "out(3)",
        composition: "replace",
      }, drawAt);
      timeline.add(drawPaths, {
        opacity: [1, 0],
        duration: 180,
        ease: "out(2)",
        composition: "replace",
      }, drawAt + DRAW_DURATION + 140);
    }

    timeline.call(function () {
      diagram.dataset.pluginArchitectureState = "complete";
    }, drawAt + DRAW_DURATION + 340);

    return timeline;
  }

  function animateDiagram(diagram) {
    var animeApi = getAnime();
    var svg = getActiveSvg(diagram);
    var timeline;

    if (diagram.dataset.pluginArchitectureState === "running" || diagram.classList.contains("is-clicked")) {
      return;
    }

    if (reduceMotion.matches || !animeApi || !svg) {
      setSeated(diagram);
      return;
    }

    timeline = buildTimeline(diagram, svg, animeApi);

    if (!timeline) {
      setSeated(diagram);
      return;
    }

    diagram.dataset.pluginArchitectureState = "running";
    timeline.play();
  }

  function armDiagram(diagram) {
    clearDiagram(diagram);

    if (reduceMotion.matches || !("IntersectionObserver" in window)) {
      setSeated(diagram);
      return;
    }

    var startTimer = null;
    var observer = new IntersectionObserver(function (entries) {
      entries.forEach(function (entry) {
        if (!entry.isIntersecting) {
          window.clearTimeout(startTimer);
          startTimer = null;
          return;
        }

        if (startTimer || diagram.dataset.pluginArchitectureState === "running" || diagram.classList.contains("is-clicked")) {
          return;
        }

        startTimer = window.setTimeout(function () {
          startTimer = null;
          observer.disconnect();
          animateDiagram(diagram);
        }, IN_VIEW_DELAY);
      });
    }, { threshold: 0.38 });

    observer.observe(diagram);
  }

  function handleResize() {
    window.clearTimeout(resizeTimer);
    resizeTimer = window.setTimeout(function () {
      toArray(document.querySelectorAll(DIAGRAM_SELECTOR)).forEach(function (diagram) {
        if (diagram.classList.contains("is-clicked")) {
          setSeated(diagram);
        } else if (diagram.dataset.pluginArchitectureState !== "running") {
          clearDiagram(diagram);
        }
      });
    }, 120);
  }

  document.addEventListener("DOMContentLoaded", function () {
    toArray(document.querySelectorAll(DIAGRAM_SELECTOR)).forEach(armDiagram);
    window.addEventListener("resize", handleResize);
  });
})();
