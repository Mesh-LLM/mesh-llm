(function () {
  'use strict';

  var reduceMotionMedia = window.matchMedia('(prefers-reduced-motion: reduce)');
  var coarsePointerMedia = window.matchMedia('(pointer: coarse)');
  var reduceMotion = reduceMotionMedia.matches;
  var section = null;
  var sticky = null;
  var nav = null;
  var heroViz = null;
  var heroTitle = null;
  var heroFooter = null;
  var finalHookLines = [];
  var finalHookBiggerWords = [];
  var finalHookPulseTimeline = null;
  var finalHookWordAnimationPlayed = false;
  var finalHookPreviousStageProgress = 0;
  var serverPlate = null;
  var tensorStack = null;
  var layoutRoot = null;
  var stage2FlightLayer = null;
  var meshNodes = [];
  var modelLayers = [];
  var stage2Layout = null;
  var sliceTimeline = null;
  var sliceCutterRaf = null;
  var sliceTimelineState = { progress: 0 };
  var sliceTimelinePainted = false;
  var layerAssignments = {};
  var layoutClearTimers = [];
  var phaseName = null;
  var phaseIndex = null;
  var scrubInput = null;
  var phasePanels = [];
  var timeline = null;
  var scrollObserver = null;
  var timelineState = { progress: 0 };
  var smoothState = { progress: 0 };
  var sectionVarCache = {};
  var scrollSmoothRaf = null;
  var scrubScrollState = null;
  var scrubScrollRaf = null;
  var scrubScrollBehaviorRestore = null;
  var scrubScrollToken = 0;
  var scrubberSeekRaf = null;
  var completedGridClipRaf = null;
  var raf = null;
  var activePhase = -1;
  var cachedServer = null;
  var cachedApertureMaxRadius = null;
  var cachedMeshGeometry = null;
  var cachedTitleSnapY = null;
  var currentMeshGeometryScale = 1;
  var currentMeshGeometryOriginX = 0;
  var currentMeshGeometryOriginY = 0;
  var currentHeroVizStageY = 0;
  var autoPlacementComplete = false;
  var placementStarted = false;
  var placementMotionActive = false;
  var placementSettleBarrier = false;
  var isScrubbing = false;
  var scrubberPointerId = null;
  var latestScrollProgress = 0;
  var usesScrollObserver = false;
  var manualStageSeekActive = false;
  var manualStageSeekScrollY = 0;
  var titleSnapState = { progress: 0 };
  var titleSnapTarget = 0;
  var titleSnapPrimed = false;
  var titleSnapAnimationTarget = null;
  var titleSnapRaf = null;
  var titleScrollProgress = 0;
  var titleFadeProgress = 0;
  var heroVizStableRequested = false;
  var STAGE_INTRO_PROGRESS = 0.16;
  // Leaves the final slice of the stage as a settled hold after the aperture closes.
  var STAGE_ANIMATION_END_PROGRESS = 0.955;
  var APERTURE_OPEN_START = 0.245;
  var APERTURE_OPEN_END = 0.380;
  var APERTURE_CLOSE_START = 1.095;
  var APERTURE_CLOSE_END = 1.153;
  var APERTURE_FADE_START = 1.125;
  var APERTURE_FADE_END = 1.155;
  var APERTURE_DETAIL_FADE_END = 1.146;
  var APERTURE_WORKBENCH_EXIT_END = 1.148;
  var APERTURE_RING_RETURN_END = 1.125;
  var APERTURE_SCRUBBER_EXIT_END = 1.142;
  var DESKTOP_APERTURE_ENTRY_SCROLL_FACTOR = 0.5;
  var FINAL_HOOK_FADE_START = 1.136;
  var FINAL_HOOK_FADE_END = 1.153;
  var FINAL_HOOK_PULSE_TRIGGER_PROGRESS = 1.154;
  var FINAL_HOOK_PULSE_SCALE = 1.1125;
  var FINAL_HOOK_PULSE_DURATION = 920;
  var FINAL_HOOK_PULSE_PEAK_DURATION = 310;
  var FINAL_HOOK_PULSE_GAP = 1500;
  var TITLE_SNAP_LEAD_START_PROGRESS = 0.035;
  var TITLE_SNAP_FORWARD_PROGRESS = 0.085;
  var TITLE_SNAP_REVERSE_PROGRESS = 0.065;
  var TOUCH_LANDSCAPE_TITLE_SNAP_REVERSE_PROGRESS = 0.155;
  var TITLE_SNAP_LEAD_PROGRESS = 0;
  var TITLE_SNAP_Y = -570;
  var TITLE_CLEAR_START_STAGE_PROGRESS = 0.205;
  var TITLE_CLEAR_END_STAGE_PROGRESS = 0.245;
  var FOOTER_CLEAR_START_STAGE_PROGRESS = 0.245;
  var FOOTER_CLEAR_END_STAGE_PROGRESS = 0.405;

  var PHASES = [
    { id: 'phase-1', label: 'materialize model', start: 0.385, end: 0.465, apply: applyPhaseOne },
    { id: 'phase-2', label: 'slice layers', start: 0.465, end: 0.575, apply: applyPhaseTwo },
    { id: 'phase-3', label: 'compact model', start: 0.575, end: 0.655, apply: applyPhaseThree },
    { id: 'phase-4', label: 'materialize nodes', start: 0.655, end: 0.775, apply: applyPhaseFour },
    { id: 'phase-5', label: 'distribute layers', start: 0.775, end: 0.955, apply: applyPhaseFive },
    { id: 'phase-6', label: 'lock split', start: 0.955, end: 1.015, apply: applyPhaseSix },
    { id: 'phase-7', label: 'align nodes', start: 1.015, end: 1.095, apply: applyPhaseSeven },
    { id: 'phase-8', label: 'aperture close', start: 1.095, end: 1.155, apply: applyPhaseEight },
  ];
  var SHRINK_PHASE_INDEX = 2;
  var NODE_PHASE_INDEX = 3;
  var MESH_PHASE_INDEX = 4;
  var LOCK_PHASE_INDEX = 5;
  var ALIGN_PHASE_INDEX = 6;
  var SCRUBBER_START = PHASES[0].start;
  var SCRUBBER_END = PHASES[PHASES.length - 1].end;
  var LEARN_MORE_TARGET_PROGRESS = SCRUBBER_END;
  var LEARN_MORE_SCRUB_SPEED_MULTIPLIER = 16;
  var scrubberTickCount = 73;
  var NODE_IDS = ['server', 'cloud', 'workstation', 'gpu', 'laptop', 'mini'];
  var DEFAULT_ASSIGNMENTS = {
    0: 'server',
    1: 'server',
    2: 'server',
    3: 'server',
    4: 'cloud',
    5: 'cloud',
    6: 'workstation',
    7: 'gpu',
    8: 'laptop',
    9: 'mini',
  };
  var NODE_CARD_OFFSETS = {
    server: { x: 650, y: 346 },
    cloud: { x: 48, y: 28 },
    workstation: { x: 76, y: 62 },
    gpu: { x: -76, y: 64 },
    laptop: { x: -165, y: -80 },
    mini: { x: 40, y: -105 },
  };
  var PLACEMENT_MOVE_DURATION = 620;
  var PLACEMENT_LAYER_START = 0;
  var PLACEMENT_LAYER_WINDOW = 0.095;
  var PLACEMENT_LAYER_END = PLACEMENT_LAYER_START + PLACEMENT_LAYER_WINDOW * 10;
  var PLACEMENT_PROGRESS_EPSILON = 0.001;
  var SCROLL_SMOOTH_SYNC = 0.9;
  var SCROLL_SMOOTH_STEP = 0.181;
  var TOUCH_SCROLL_SMOOTH_SYNC = 1;
  var TOUCH_SCROLL_SMOOTH_STEP = 0.44;
  var TOUCH_MODEL_MIN_SCALE = 0.018;
  var TOUCH_PLACEMENT_SNAP_PROGRESS = 0.12;

  function clamp(value, min, max) {
    return Math.min(Math.max(value, min), max);
  }

  function useTouchScrollProfile() {
    return coarsePointerMedia.matches;
  }

  function scrollSmoothSync() {
    return useTouchScrollProfile() ? TOUCH_SCROLL_SMOOTH_SYNC : SCROLL_SMOOTH_SYNC;
  }

  function scrollSmoothStep() {
    return useTouchScrollProfile() ? TOUCH_SCROLL_SMOOTH_STEP : SCROLL_SMOOTH_STEP;
  }

  function useLiteStage2Motion() {
    return useTouchScrollProfile();
  }

  function useTouchLandscapeStage() {
    var width = sticky ? sticky.clientWidth : window.innerWidth;
    var height = sticky ? sticky.clientHeight : window.innerHeight;
    return useTouchScrollProfile() && width > height;
  }

  function titleSnapReverseProgress() {
    return useTouchLandscapeStage() ? TOUCH_LANDSCAPE_TITLE_SNAP_REVERSE_PROGRESS : TITLE_SNAP_REVERSE_PROGRESS;
  }

  function useLandscapeStage() {
    var width = sticky ? sticky.clientWidth : window.innerWidth;
    var height = sticky ? sticky.clientHeight : window.innerHeight;
    return width > height;
  }

  function landscapeVizStageY() {
    if (!sticky || !useLandscapeStage()) return 0;
    return clamp(sticky.clientHeight * 0.18, 150, 190);
  }

  function shortLandscapeVizBaseY() {
    if (!sticky || !useLandscapeStage() || sticky.clientHeight > 480) return 0;
    return -clamp(sticky.clientHeight * 0.32, 90, 120);
  }

  function usePhonePortraitStage() {
    var width = sticky ? sticky.clientWidth : window.innerWidth;
    var height = sticky ? sticky.clientHeight : window.innerHeight;
    return width <= 640 && height > width;
  }

  function phonePortraitVizStageY() {
    if (!sticky || !usePhonePortraitStage()) return 0;
    return clamp(sticky.clientHeight * 0.16, 96, 150);
  }

  function phonePortraitTitleLift() {
    if (!sticky || !usePhonePortraitStage()) return 0;
    return clamp(sticky.clientHeight * 0.055, 34, 58);
  }

  function heroVizStageY() {
    return Math.max(landscapeVizStageY(), phonePortraitVizStageY());
  }

  function heroVizBaseStageY() {
    return shortLandscapeVizBaseY();
  }

  function titleTextRect() {
    if (!heroTitle) return null;

    var heading = heroTitle.querySelector('.hero-title-heading');
    var subtitle = heroTitle.querySelector('.hero-title-subtitle');
    var headingRect = heading ? heading.getBoundingClientRect() : null;
    var subtitleRect = subtitle ? subtitle.getBoundingClientRect() : null;

    if (!headingRect && !subtitleRect) return heroTitle.getBoundingClientRect();
    if (!headingRect) return subtitleRect;
    if (!subtitleRect) return headingRect;

    return {
      left: Math.min(headingRect.left, subtitleRect.left),
      right: Math.max(headingRect.right, subtitleRect.right),
      top: Math.min(headingRect.top, subtitleRect.top),
      bottom: Math.max(headingRect.bottom, subtitleRect.bottom),
      width: Math.max(headingRect.right, subtitleRect.right) - Math.min(headingRect.left, subtitleRect.left),
      height: Math.max(headingRect.bottom, subtitleRect.bottom) - Math.min(headingRect.top, subtitleRect.top),
    };
  }

  function horizontallyOverlaps(a, b, margin) {
    return a.left - margin < b.right && a.right + margin > b.left;
  }

  function diagramBlockingTop(titleRect, stageRect) {
    if (!heroViz || !titleRect || !stageRect) return null;

    var blockers = heroViz.querySelectorAll('.node-plate, .mesh-edge, .mesh-edge-trace');
    var margin = clamp((sticky ? sticky.clientWidth : window.innerWidth) * 0.018, 18, 34);
    var top = null;

    blockers.forEach(function (blocker) {
      var style = window.getComputedStyle(blocker);
      var rect;

      if (style.display === 'none' || style.visibility === 'hidden' || Number.parseFloat(style.opacity) <= 0.02) return;

      rect = blocker.getBoundingClientRect();
      if (rect.width < 1 || rect.height < 1) return;
      if (!horizontallyOverlaps(titleRect, rect, margin)) return;

      top = top === null ? rect.top - stageRect.top : Math.min(top, rect.top - stageRect.top);
    });

    return top;
  }

  function titleSnapY() {
    if (!heroTitle || !sticky) return TITLE_SNAP_Y;
    if (cachedTitleSnapY !== null) return cachedTitleSnapY;

    var server = syncServerGeometry();
    var stageRect = sticky.getBoundingClientRect();
    var stageHeight = sticky ? sticky.clientHeight : window.innerHeight;
    var titleHeight = heroTitle.offsetHeight || 124;
    var titleTop = heroTitle.offsetTop || 0;
    var textRect = titleTextRect();
    var blockerTop = diagramBlockingTop(textRect, stageRect);
    var remainingVizStageY = heroVizStageY() - currentHeroVizStageY;
    var safeTop = clamp(stageHeight * 0.08, 48, 96);
    var safeBottom = Math.max(safeTop, stageHeight - titleHeight - clamp(stageHeight * 0.05, 24, 56));
    var diagramGap = clamp(stageHeight * 0.06, 40, 76);
    var serverAnchoredTop = server.y + remainingVizStageY - server.radius - diagramGap - titleHeight;
    var diagramAnchoredTop = blockerTop === null ? serverAnchoredTop : blockerTop + remainingVizStageY - diagramGap - titleHeight;
    var targetTop = clamp(Math.min(serverAnchoredTop, diagramAnchoredTop), safeTop, safeBottom);

    targetTop = Math.max(24, targetTop - phonePortraitTitleLift());
    cachedTitleSnapY = Math.round(targetTop - titleTop);
    return cachedTitleSnapY;
  }

  function lerp(a, b, progress) {
    return a + (b - a) * progress;
  }

  function roundEven(value) {
    return Math.max(2, Math.round(value / 2) * 2);
  }

  function ramp(value, start, end) {
    if (end === start) return value >= end ? 1 : 0;
    return clamp((value - start) / (end - start), 0, 1);
  }

  function easeOut(value) {
    return 1 - Math.pow(1 - clamp(value, 0, 1), 3);
  }

  function easeInOut(value) {
    value = clamp(value, 0, 1);
    return value < 0.5 ? 4 * value * value * value : 1 - Math.pow(-2 * value + 2, 3) / 2;
  }

  function easeOutQuart(value) {
    value = clamp(value, 0, 1);
    return 1 - Math.pow(1 - value, 4);
  }

  function easeOutExpo(value) {
    value = clamp(value, 0, 1);
    return value === 1 ? 1 : 1 - Math.pow(2, -10 * value);
  }

  function fastCloseEase(value) {
    return clamp(value, 0, 1);
  }

  function setVar(name, value) {
    var nextValue = typeof value === 'number' ? String(Math.round(value * 10000) / 10000) : value;

    if (sectionVarCache[name] === nextValue) return;
    sectionVarCache[name] = nextValue;
    section.style.setProperty(name, nextValue);
  }

  function clearVar(name) {
    if (!section) return;

    delete sectionVarCache[name];
    section.style.removeProperty(name);
  }

  function syncHeroChromeMetrics() {
    var chromeHeight = nav ? nav.getBoundingClientRect().height : 0;

    if (!chromeHeight) chromeHeight = 64;

    chromeHeight = Math.round(chromeHeight * 100) / 100;
    setVar('--hero-initial-chrome', chromeHeight + 'px');
    setVar('--hero-sticky-top', chromeHeight + 'px');
  }

  function syncCompletedGridClip() {
    if (!section) return;
    var heroBottom = section.getBoundingClientRect().bottom;
    var clipBottom = clamp(heroBottom, 0, window.innerHeight);
    setVar('--stage2-grid-clip-bottom', Math.round(clipBottom) + 'px');
  }

  function requestCompletedGridClip() {
    if (!section || !document.body.classList.contains('is-stage2-complete')) return;
    if (completedGridClipRaf) return;

    completedGridClipRaf = window.requestAnimationFrame(function () {
      completedGridClipRaf = null;
      syncCompletedGridClip();
    });
  }

  function readSectionNumber(name, fallback) {
    var value = Number.parseFloat(section && section.style.getPropertyValue(name));
    return Number.isFinite(value) ? value : fallback;
  }

  function animeEase(name, fallback) {
    var ease = window.anime && window.anime.eases && window.anime.eases[name];
    return typeof ease === 'function' ? ease : fallback;
  }

  function titleSnapSpring() {
    if (!window.anime || !window.anime.spring) return null;
    return window.anime.spring({
      stiffness: 500,
      damping: 30,
      mass: 1,
      velocity: 0,
    });
  }

  function titleLeadProgress(progress) {
    var lead = ramp(progress, TITLE_SNAP_LEAD_START_PROGRESS, TITLE_SNAP_FORWARD_PROGRESS);
    return TITLE_SNAP_LEAD_PROGRESS * animeEase('outQuad', easeOut)(lead);
  }

  function applyTitleSnapTransform() {
    if (!heroTitle) return;
    var y = Math.round(titleSnapY() * titleSnapState.progress);
    var shrink = lerp(1, 0.965, animeEase('outSine', easeOut)(titleFadeProgress));
    heroTitle.style.transform = 'translateY(' + y + 'px) scale(' + Math.round(shrink * 10000) / 10000 + ')';
    applyHeroVizTransform(currentMeshGeometryScale || 1, false);
  }

  function heroVizTransform(stageY, meshScale) {
    return 'translateX(-50%) translateY(' + stageY + 'px) scale(' + meshScale + ')';
  }

  function applyHeroVizTransform(meshScale, syncGeometry) {
    var heroVizLeft;
    var originX;
    var originY;
    var server;

    if (!heroViz) return;
    syncGeometry = syncGeometry !== false;

    currentMeshGeometryScale = meshScale;
    currentHeroVizStageY = Math.round(lerp(heroVizBaseStageY(), heroVizStageY(), titleSnapState.progress));

    if (!syncGeometry) {
      heroViz.style.transform = heroVizTransform(currentHeroVizStageY, meshScale);
      return;
    }

    if (!cachedServer) {
      heroViz.style.transform = heroVizTransform(currentHeroVizStageY, 1);
    }
    server = syncServerGeometry();
    heroVizLeft = heroViz.offsetLeft - heroViz.offsetWidth / 2;
    originX = server.baseX - heroVizLeft;
    originY = server.baseY - heroViz.offsetTop;
    currentMeshGeometryOriginX = server.baseX;
    currentMeshGeometryOriginY = server.baseY;
    heroViz.style.transformOrigin = Math.round(originX * 100) / 100 + 'px ' + Math.round(originY * 100) / 100 + 'px';
    heroViz.style.transform = heroVizTransform(currentHeroVizStageY, meshScale);
    syncServerGeometry();
    syncMeshGeometry();
  }

  function cancelTitleSnapAnimation() {
    if (titleSnapRaf) window.cancelAnimationFrame(titleSnapRaf);
    titleSnapRaf = null;
    titleSnapAnimationTarget = null;
  }

  function animateTitleSnapTo(progress, target) {
    var from = titleSnapState.progress;
    var to = clamp(progress, 0, 1);
    var spring = titleSnapSpring();
    var duration = spring && spring.settlingDuration ? spring.settlingDuration : 500;
    var ease = spring && spring.ease ? spring.ease : easeOut;
    var start = 0;

    cancelTitleSnapAnimation();
    titleSnapAnimationTarget = target;

    if (!target) {
      function follow() {
        var desired = titleLeadProgress(titleScrollProgress);
        var delta = desired - titleSnapState.progress;
        var followAmount = clamp(ease(0.085), 0.08, 0.34);

        titleSnapState.progress += delta * followAmount;
        if (Math.abs(delta) < 0.0015) titleSnapState.progress = desired;
        applyTitleSnapTransform();

        if (titleSnapState.progress === desired) {
          titleSnapRaf = null;
          titleSnapAnimationTarget = null;
          return;
        }

        titleSnapRaf = window.requestAnimationFrame(follow);
      }

      titleSnapRaf = window.requestAnimationFrame(follow);
      return;
    }

    function tick(now) {
      if (!start) start = now;

      var localProgress = clamp((now - start) / duration, 0, 1);
      titleSnapState.progress = lerp(from, to, ease(localProgress));
      applyTitleSnapTransform();

      if (localProgress < 1) {
        titleSnapRaf = window.requestAnimationFrame(tick);
        return;
      }

      titleSnapRaf = null;
      titleSnapAnimationTarget = null;
      titleSnapState.progress = to;
      applyTitleSnapTransform();
    }

    titleSnapRaf = window.requestAnimationFrame(tick);
  }

  function setTitleSnapTarget(target, immediate) {
    target = target ? 1 : 0;
    if (!heroTitle) return;

    var targetProgress = target ? 1 : titleLeadProgress(titleScrollProgress);

    if (immediate || reduceMotion) {
      cancelTitleSnapAnimation();
      titleSnapTarget = target;
      titleSnapState.progress = targetProgress;
      applyTitleSnapTransform();
      return;
    }

    if (target === titleSnapTarget) return;
    titleSnapTarget = target;
    animateTitleSnapTo(targetProgress, target);
  }

  function syncTitleSnap(progress) {
    var target = titleSnapTarget;
    titleScrollProgress = progress;

    if (!titleSnapPrimed) {
      titleSnapPrimed = true;
      setTitleSnapTarget(progress >= TITLE_SNAP_FORWARD_PROGRESS ? 1 : 0, true);
      return;
    }

    if (titleSnapTarget === 0 && progress >= TITLE_SNAP_FORWARD_PROGRESS) target = 1;
    if (titleSnapTarget === 1 && progress <= titleSnapReverseProgress()) target = 0;
    setTitleSnapTarget(target, false);

    if (target === 0 && titleSnapAnimationTarget === null && progress < TITLE_SNAP_FORWARD_PROGRESS) {
      titleSnapState.progress = titleLeadProgress(progress);
      applyTitleSnapTransform();
    }
  }

  function syncTitleFromProgress(titleProgress, stageProgress) {
    if (!heroTitle || !section) return;

    var heroReady = document.body.classList.contains('is-hero-complete') || !document.body.classList.contains('home-intro');
    if (titleProgress > 0.01) {
      document.body.classList.add('is-hero-complete');
      document.body.classList.remove('home-intro');
      heroReady = true;
    }
    if (!heroReady && titleProgress < 0.001) return;

    var titleOut = ramp(stageProgress, TITLE_CLEAR_START_STAGE_PROGRESS, TITLE_CLEAR_END_STAGE_PROGRESS);
    titleFadeProgress = easeOut(titleOut);
    section.classList.toggle('is-stage2-title-cleared', titleFadeProgress > 0.985);
    syncTitleSnap(titleProgress);
    heroTitle.style.opacity = String(Math.round((1 - titleFadeProgress) * 10000) / 10000);
    applyTitleSnapTransform();
  }

  function syncTitleFromScroll() {
    var titleProgress = progressFromScroll();
    syncTitleFromProgress(titleProgress, stageProgressFromScrollProgress(titleProgress));
  }

  function progressFromScrubberProgress(progress) {
    return lerp(SCRUBBER_START, SCRUBBER_END, clamp(progress, 0, 1));
  }

  function scrubberProgressFromProgress(progress) {
    return ramp(progress, SCRUBBER_START, SCRUBBER_END);
  }

  function snapScrubberProgressToTick(progress) {
    var segments = Math.max(1, scrubberTickCount - 1);
    return Math.round(clamp(progress, 0, 1) * segments) / segments;
  }

  function tickPosition(progress) {
    var snapped = snapScrubberProgressToTick(progress);
    return Math.round(snapped * (scrubberTickCount - 1)) * 4 + 'px';
  }

  function phaseForProgress(progress) {
    var i;

    for (i = PHASES.length - 1; i >= 0; i -= 1) {
      if (progress >= PHASES[i].start) return i;
    }

    return 0;
  }

  function phaseProgress(phase, progress) {
    return ramp(progress, phase.start, phase.end);
  }

  function titlePresence(stageProgress, enterStart, enterEnd, exitStart, exitEnd) {
    return clamp(easeOut(ramp(stageProgress, enterStart, enterEnd)) - easeInOut(ramp(stageProgress, exitStart, exitEnd)), 0, 1);
  }

  function phaseTitleModelOffset(titleOpacity) {
    var width = window.innerWidth || (sticky ? sticky.clientWidth : 1440);
    var height = window.innerHeight || (sticky ? sticky.clientHeight : 900);
    var offset = 64;

    if (height <= 480 && width > height) {
      offset = 74;
    } else if (width <= 640) {
      offset = 68;
      if (height <= 760) offset = 84;
      if (height <= 650) offset = 98;
      if (width <= 380) offset += 8;
    } else if (height <= 780) {
      offset = 92;
    }

    return Math.round(offset * clamp(titleOpacity, 0, 1));
  }

  function finalHookWordText(word) {
    return (word.textContent || '').trim().toLowerCase().replace(/[^a-z0-9]+/g, '');
  }

  function fallbackSplitFinalHookLineWords(line) {
    var parts = (line.textContent || '').split(/(\s+)/);
    var words = [];

    line.textContent = '';
    parts.forEach(function (part) {
      var word;

      if (!part) return;
      if (/^\s+$/.test(part)) {
        line.appendChild(document.createTextNode(part));
        return;
      }

      word = document.createElement('span');
      word.className = 'stage2-final-word';
      word.textContent = part;
      line.appendChild(word);
      words.push(word);
    });

    return words;
  }

  function splitFinalHookLineWords(line) {
    var split = null;
    var words = [];

    if (!line) return words;
    line.setAttribute('data-stage2-final-line-text', (line.textContent || '').trim());

    if (window.anime && typeof window.anime.splitText === 'function') {
      try {
        split = window.anime.splitText(line, {
          words: { class: 'stage2-final-word' },
          accessible: false,
        });
        if (split && split.words) words = Array.prototype.slice.call(split.words);
      } catch (error) {
        words = [];
      }
    }

    if (!words.length) words = fallbackSplitFinalHookLineWords(line);

    words.forEach(function (word) {
      word.setAttribute('data-stage2-final-word', '');
      if (finalHookWordText(word) === 'bigger') {
        word.setAttribute('data-stage2-final-keyword', 'bigger');
        finalHookBiggerWords.push(word);
      }
    });

    return words;
  }

  function initFinalHookWords() {
    finalHookBiggerWords = [];
    finalHookLines.forEach(splitFinalHookLineWords);
  }

  function clearFinalHookPulseTimeline() {
    if (finalHookPulseTimeline && typeof finalHookPulseTimeline.pause === 'function') {
      finalHookPulseTimeline.pause();
    }
    finalHookPulseTimeline = null;
  }

  function setFinalHookPulseClass(word, active) {
    if (!word) return;
    word.classList.toggle('is-stage2-final-word-pulsing', active);
    if (active) word.style.transform = 'scale(1)';
  }

  function setFinalHookWordGlow(word, state) {
    if (!word || !state) return;
    word.style.setProperty('--stage2-final-word-glow', String(Math.round(state.glow * 1000) / 1000));
    word.style.setProperty('--stage2-final-word-core', String(Math.round(state.core * 1000) / 1000));
    word.style.setProperty('--stage2-final-word-glow-x', Math.round(state.sweep * 100) + '%');
    word.style.setProperty('--stage2-final-word-glow-stretch', String(Math.round(state.stretch * 1000) / 1000));
  }

  function clearFinalHookWordGlow(word) {
    if (!word) return;
    word.style.removeProperty('--stage2-final-word-glow');
    word.style.removeProperty('--stage2-final-word-core');
    word.style.removeProperty('--stage2-final-word-glow-x');
    word.style.removeProperty('--stage2-final-word-glow-stretch');
  }

  function clearFinalHookWordPulseStyles() {
    finalHookBiggerWords.forEach(function (word) {
      word.classList.remove('is-stage2-final-word-pulsing');
      word.style.transform = '';
      clearFinalHookWordGlow(word);
    });
  }

  function resetFinalHookWordPulse() {
    finalHookWordAnimationPlayed = false;
    clearFinalHookPulseTimeline();
    if (window.anime && typeof window.anime.remove === 'function') {
      window.anime.remove(finalHookBiggerWords);
    }
    clearFinalHookWordPulseStyles();
  }

  function addFinalHookWordPulse(timeline, word, start) {
    var releaseDuration = FINAL_HOOK_PULSE_DURATION - FINAL_HOOK_PULSE_PEAK_DURATION;
    var riseEase = 'out(4.2)';
    var releaseEase = 'inOut(2.8)';

    if (!timeline || !word) return;

    timeline.add(word, {
      scale: [
        { to: FINAL_HOOK_PULSE_SCALE, duration: FINAL_HOOK_PULSE_PEAK_DURATION, ease: riseEase },
        { to: 1, duration: releaseDuration, ease: releaseEase },
      ],
      '--stage2-final-word-glow-x': ['0%', '100%'],
      '--stage2-final-word-glow': [
        { to: 1, duration: FINAL_HOOK_PULSE_PEAK_DURATION, ease: riseEase },
        { to: 0, duration: releaseDuration, ease: releaseEase },
      ],
      '--stage2-final-word-core': [
        { to: 1, duration: FINAL_HOOK_PULSE_PEAK_DURATION, ease: riseEase },
        { to: 0, duration: releaseDuration, ease: releaseEase },
      ],
      '--stage2-final-word-glow-stretch': [
        { to: 1.34, duration: FINAL_HOOK_PULSE_PEAK_DURATION, ease: riseEase },
        { to: 0.98, duration: releaseDuration, ease: releaseEase },
      ],
      duration: FINAL_HOOK_PULSE_DURATION,
      ease: 'inOut(6)',
      composition: 'replace',
      onBegin: function () {
        setFinalHookPulseClass(word, true);
        setFinalHookWordGlow(word, {
          glow: 0,
          core: 0,
          sweep: 0,
          stretch: 0.88,
        });
      },
      onComplete: function () {
        setFinalHookPulseClass(word, false);
        word.style.transform = '';
        clearFinalHookWordGlow(word);
      },
    }, start);
  }

  function playFinalHookWordPulseTimeline() {
    var topWord = finalHookBiggerWords[0];
    var bottomWord = finalHookBiggerWords[1];
    var secondPulseStart = FINAL_HOOK_PULSE_DURATION + FINAL_HOOK_PULSE_GAP;

    if (finalHookWordAnimationPlayed || !topWord || !bottomWord) return;

    finalHookWordAnimationPlayed = true;
    if (reduceMotion || !window.anime || typeof window.anime.createTimeline !== 'function') return;

    clearFinalHookPulseTimeline();
    if (typeof window.anime.remove === 'function') window.anime.remove(finalHookBiggerWords);
    clearFinalHookWordPulseStyles();

    finalHookPulseTimeline = window.anime.createTimeline({
      autoplay: false,
      defaults: { ease: 'out(3)' },
      onComplete: function () {
        finalHookPulseTimeline = null;
        clearFinalHookWordPulseStyles();
      },
    });

    addFinalHookWordPulse(finalHookPulseTimeline, topWord, 0);
    addFinalHookWordPulse(finalHookPulseTimeline, bottomWord, secondPulseStart);
    finalHookPulseTimeline.play();
  }

  function syncFinalHookWordPulse(stageProgress) {
    var crossedForward = finalHookPreviousStageProgress < FINAL_HOOK_PULSE_TRIGGER_PROGRESS && stageProgress >= FINAL_HOOK_PULSE_TRIGGER_PROGRESS;

    if (crossedForward) {
      playFinalHookWordPulseTimeline();
    } else if (stageProgress < FINAL_HOOK_PULSE_TRIGGER_PROGRESS - 0.012) {
      finalHookWordAnimationPlayed = false;
    }

    if (stageProgress <= FINAL_HOOK_FADE_START - 0.02) resetFinalHookWordPulse();
    finalHookPreviousStageProgress = stageProgress;
  }

  function setStage2TitleProgress(stageProgress) {
    var split = titlePresence(stageProgress, 0.486, 0.522, 0.622, 0.662);
    var nodes = titlePresence(stageProgress, 0.666, 0.704, 0.754, 0.810);
    var place = titlePresence(stageProgress, 0.794, 0.836, 0.948, 1.012);
    var phaseTitle = Math.max(split, nodes, place);
    var final = easeOut(ramp(stageProgress, FINAL_HOOK_FADE_START, FINAL_HOOK_FADE_END));

    setVar('--stage2-title-split-opacity', Math.round(split * 10000) / 10000);
    setVar('--stage2-title-nodes-opacity', Math.round(nodes * 10000) / 10000);
    setVar('--stage2-title-place-opacity', Math.round(place * 10000) / 10000);
    setVar('--stage2-title-final-opacity', Math.round(final * 10000) / 10000);
    setVar('--stage2-title-model-offset', phaseTitleModelOffset(phaseTitle) + 'px');
    syncFinalHookWordPulse(stageProgress);
    if (section) section.classList.toggle('is-stage2-final-hook-visible', final > 0.02);
  }

  function modelTargetSize() {
    var width = sticky ? sticky.clientWidth : window.innerWidth;
    var height = window.innerHeight || (sticky ? sticky.clientHeight : 720);

    if (width <= 640) {
      return {
        width: Math.min(width * 0.88, 430),
        height: Math.max(420, Math.min(height * 0.72, 520)),
      };
    }

    return {
      width: Math.min(width * 0.72, 860),
      height: Math.max(560, Math.min(height * 0.72, 660)),
    };
  }

  function connectorProgress(fromPhaseIndex, progress) {
    var from = PHASES[fromPhaseIndex];
    var to = PHASES[fromPhaseIndex + 1];
    if (!from || !to) return 0;

    return ramp(progress, from.end, to.start);
  }

  function syncPhaseState(progress) {
    var nextPhase = phaseForProgress(progress);
    if (nextPhase === activePhase) return;

    activePhase = nextPhase;
    var phase = PHASES[nextPhase];
    if (phaseName) phaseName.textContent = phase.label;
    if (phaseIndex) phaseIndex.textContent = String(nextPhase + 1).padStart(2, '0') + ' / ' + String(PHASES.length).padStart(2, '0');
    section.setAttribute('data-stage2-phase', phase.id);

    if (scrubInput) {
      scrubInput.setAttribute('aria-valuetext', phase.label);
    }
  }

  function applyPhaseOne(context) {
    var progress = clamp(context.localProgress, 0, 1);
    var expandEase = animeEase('outQuart', function (value) {
      value = clamp(value, 0, 1);
      return 1 - Math.pow(1 - value, 4);
    });
    var settleEase = animeEase('outSine', easeOut);
    var materialEase = animeEase('outQuad', easeOut);
    var expand = expandEase(ramp(progress, 0, 0.64));
    var settle = settleEase(ramp(progress, 0.58, 1));
    var material = materialEase(ramp(progress, 0.08, 0.72));
    var target = modelTargetSize();
    var sizeMultiplier = lerp(0.012, 1.045, expand);
    var width = target.width * sizeMultiplier;
    var height = target.height * sizeMultiplier;
    var y = lerp(22, -12, expand);

    width = lerp(width, target.width, settle);
    height = lerp(height, target.height, settle);
    y = lerp(y, 0, settle);

    setVar('--stage2-phase-1', progress);
    setVar('--stage2-model-opacity', easeOut(ramp(progress, 0.015, 0.22)));
    if (useLiteStage2Motion()) {
      setVar('--stage2-model-width', roundEven(target.width) + 'px');
      setVar('--stage2-model-height', roundEven(target.height) + 'px');
      setVar('--stage2-model-scale', Math.max(TOUCH_MODEL_MIN_SCALE, sizeMultiplier));
    } else {
      setVar('--stage2-model-width', roundEven(width) + 'px');
      setVar('--stage2-model-height', roundEven(height) + 'px');
      setVar('--stage2-model-scale', 1);
    }
    setVar('--stage2-model-y', Math.round(y) + 'px');
    setVar('--stage2-model-blur', useLiteStage2Motion() ? '0px' : Math.round(lerp(7, 0, easeOut(ramp(progress, 0.06, 0.46))) * 100) / 100 + 'px');
  }

  function applyPhaseTwo(context) {
    var progress = clamp(context.localProgress, 0, 1);
    var slice = ramp(progress, 0.02, 0.98);

    setVar('--stage2-phase-2', progress);
    syncSliceTimeline(slice);
    section.classList.toggle('is-stage2-slicing', progress > 0.02);
  }

  function applyPhaseThree(context) {
    var progress = clamp(context.localProgress, 0, 1);

    setVar('--stage2-phase-3', progress);
  }

  function applyPhaseFour(context) {
    var progress = clamp(context.localProgress, 0, 1);
    var nodeProgress = easeOutExpo(ramp(progress, 0.04, 0.30));

    setVar('--stage2-phase-4', progress);
    setVar('--stage2-node-progress', nodeProgress);
  }

  function applyPhaseFive(context) {
    var progress = clamp(context.localProgress, 0, 1);
    var placement = easeInOut(ramp(progress, PLACEMENT_LAYER_START, PLACEMENT_LAYER_END));

    setVar('--stage2-phase-5', progress);
    setVar('--stage2-placement-progress', placement);
  }

  function applyPhaseSix(context) {
    var progress = clamp(context.localProgress, 0, 1);
    var lockCard = easeOutQuart(ramp(progress, 0, 0.52));
    var lockMark = easeOutQuart(ramp(progress, 0.10, 0.46));
    var lockCheck = easeOutQuart(ramp(progress, 0.18, 0.54));
    var lockText = easeOutQuart(ramp(progress, 0.30, 0.74));
    var lockRoute = easeOutQuart(ramp(progress, 0.42, 0.86));
    var lockNodes = easeOutQuart(ramp(progress, 0.56, 0.96));
    var lock = Math.max(lockMark, lockText, lockRoute);

    setVar('--stage2-phase-6', progress);
    if (!isPlacementSettled()) {
      setVar('--stage2-lock-card-progress', 0);
      setVar('--stage2-lock-progress', 0);
      setVar('--stage2-lock-mark-progress', 0);
      setVar('--stage2-lock-check-progress', 0);
      setVar('--stage2-lock-text-progress', 0);
      setVar('--stage2-lock-route-progress', 0);
      setVar('--stage2-lock-node-progress', 0);
      return;
    }

    setVar('--stage2-lock-card-progress', clamp(lockCard, 0, 1));
    setVar('--stage2-lock-progress', clamp(lock, 0, 1));
    setVar('--stage2-lock-mark-progress', clamp(lockMark, 0, 1));
    setVar('--stage2-lock-check-progress', clamp(lockCheck, 0, 1));
    setVar('--stage2-lock-text-progress', clamp(lockText, 0, 1));
    setVar('--stage2-lock-route-progress', clamp(lockRoute, 0, 1));
    setVar('--stage2-lock-node-progress', clamp(lockNodes, 0, 1));
    if (context.progress >= context.phase.start) setVar('--stage2-placement-progress', 1);
  }

  function applyPhaseSeven(context) {
    var progress = clamp(context.localProgress, 0, 1);
    var collapse = isPlacementSettled() ? easeInOut(ramp(progress, 0.08, 0.92)) : 0;

    setVar('--stage2-phase-7', progress);
    setVar('--stage2-collapse-progress', collapse);
    if (context.progress >= context.phase.start && isPlacementSettled()) {
      setVar('--stage2-lock-card-progress', 1);
      setVar('--stage2-lock-progress', clamp(1 - ramp(progress, 0.42, 0.86), 0, 1));
      setVar('--stage2-lock-mark-progress', clamp(1 - ramp(progress, 0.42, 0.86), 0, 1));
      setVar('--stage2-lock-check-progress', clamp(1 - ramp(progress, 0.42, 0.86), 0, 1));
      setVar('--stage2-lock-text-progress', clamp(1 - ramp(progress, 0.30, 0.74), 0, 1));
      setVar('--stage2-lock-route-progress', clamp(1 - ramp(progress, 0.34, 0.78), 0, 1));
      setVar('--stage2-lock-node-progress', clamp(1 - ramp(progress, 0.38, 0.82), 0, 1));
    }
  }

  function applyPhaseEight(context) {
    setVar('--stage2-phase-8', context.localProgress);
  }

  function applyPhaseScaffold(stageProgress) {
    PHASES.forEach(function (phase, index) {
      var localProgress = phaseProgress(phase, stageProgress);
      var panel = section.querySelector('[data-stage2-phase-panel="' + phase.id + '"]');

      if (panel) {
        panel.style.opacity = String(index === activePhase ? 1 : 0);
        panel.style.pointerEvents = 'none';
      }

      phase.apply({
        progress: stageProgress,
        localProgress: localProgress,
        index: index,
        phase: phase,
        previousProgress: index > 0 ? phaseProgress(PHASES[index - 1], stageProgress) : 0,
        nextProgress: index < PHASES.length - 1 ? phaseProgress(PHASES[index + 1], stageProgress) : 0,
      });

      if (index < PHASES.length - 1) {
        setVar('--stage2-phase-' + (index + 1) + '-to-' + (index + 2), connectorProgress(index, stageProgress));
      }
    });
  }

  function invalidateGeometry() {
    cachedServer = null;
    cachedApertureMaxRadius = null;
    cachedMeshGeometry = null;
    cachedTitleSnapY = null;
  }

  function setElementVar(element, name, value) {
    var nextValue;

    if (!element) return;

    nextValue = typeof value === 'number' ? String(Math.round(value * 10000) / 10000) : value;
    element.__stage2VarCache = element.__stage2VarCache || {};
    if (element.__stage2VarCache[name] === nextValue) return;
    element.__stage2VarCache[name] = nextValue;
    element.style.setProperty(name, nextValue);
  }

  function clearElementVar(element, name) {
    if (!element) return;

    if (element.__stage2VarCache) delete element.__stage2VarCache[name];
    element.style.removeProperty(name);
  }

  function nodeCardSize(id, scale) {
    var width = sticky ? sticky.clientWidth : window.innerWidth;
    var mobile = width <= 640;
    var compact = width <= 900;
    var baseWidth = mobile ? 116 : compact ? 132 : 152;
    var baseHeight = mobile ? 86 : compact ? 100 : 116;
    var capacityBoost = id === 'server' ? 30 : id === 'cloud' ? 16 : id === 'workstation' ? 10 : 0;
    var layerCount = nodeTargetLayerCount(id);
    var rowHeight = mobile ? 22 : 24;
    var rowGap = 5;
    var hostChrome = mobile ? 84 : 96;
    var requiredHeight = hostChrome + (layerCount * rowHeight) + (Math.max(0, layerCount - 1) * rowGap);

    return {
      width: Math.max(Math.round((baseWidth + capacityBoost) * scale), mobile ? 112 : 124),
      height: Math.max(Math.round((baseHeight + capacityBoost * 0.42) * scale), requiredHeight),
    };
  }

  function nodeTargetLayerCount(id) {
    var count = 0;
    Object.keys(DEFAULT_ASSIGNMENTS).forEach(function (layerId) {
      if (DEFAULT_ASSIGNMENTS[layerId] === id) count += 1;
    });
    return count;
  }

  function nodeCardOffset(id) {
    var width = sticky ? sticky.clientWidth : window.innerWidth;
    var factor = width <= 640 ? 0.46 : width <= 900 ? 0.72 : 1;
    var offset = NODE_CARD_OFFSETS[id] || { x: 0, y: 0 };
    var serverReturn = id === 'server' ? readSectionNumber('--stage2-server-return-progress', 0) : 0;
    var returnFactor = id === 'server' ? 1 - clamp(serverReturn, 0, 1) : 1;

    return {
      x: offset.x * factor * returnFactor,
      y: offset.y * factor * returnFactor,
    };
  }

  function transformMeshNodeGeometry(item) {
    var scale = currentMeshGeometryScale || 1;

    if (!item) return;

    item.x = currentMeshGeometryOriginX + ((item.baseX - currentMeshGeometryOriginX) * scale);
    item.y = currentHeroVizStageY + currentMeshGeometryOriginY + ((item.baseY - currentMeshGeometryOriginY) * scale);
    item.radius = item.baseRadius * scale;
  }

  function writeMeshNodeGeometry(id, item) {
    var offset;
    var cardX;
    var cardY;
    var scale;

    if (!item || !item.stage2Node) return;

    transformMeshNodeGeometry(item);
    scale = Number.parseFloat(item.stage2Node.style.getPropertyValue('--stage2-node-scale')) || 1;
    item.size = nodeCardSize(id, scale);

    offset = nodeCardOffset(id);
    cardX = item.x + offset.x;
    cardY = item.y + offset.y;

    setElementVar(item.stage2Node, '--stage2-node-x', Math.round(item.x * 100) / 100 + 'px');
    setElementVar(item.stage2Node, '--stage2-node-y', Math.round(item.y * 100) / 100 + 'px');
    setElementVar(item.stage2Node, '--stage2-node-r', Math.round(item.radius * 100) / 100 + 'px');
    setElementVar(item.stage2Node, '--stage2-node-card-x', Math.round(cardX * 100) / 100 + 'px');
    setElementVar(item.stage2Node, '--stage2-node-card-y', Math.round(cardY * 100) / 100 + 'px');
    setElementVar(item.stage2Node, '--stage2-node-card-w', item.size.width + 'px');
    setElementVar(item.stage2Node, '--stage2-node-card-h', item.size.height + 'px');

    item.cardX = cardX;
    item.cardY = cardY;
  }

  function syncMeshGeometry() {
    if (!sticky) return {};

    if (cachedMeshGeometry) {
      NODE_IDS.forEach(function (id) {
        writeMeshNodeGeometry(id, cachedMeshGeometry[id]);
      });
      return cachedMeshGeometry;
    }

    var stageRect = sticky.getBoundingClientRect();
    var geometry = {};
    var transformScale = currentMeshGeometryScale || 1;

    NODE_IDS.forEach(function (id) {
      var stageNode = section.querySelector('.hero-viz [data-node-id="' + id + '"] .node-plate');
      var stage2Node = section.querySelector('[data-stage2-node="' + id + '"]');
      var nodeRect;
      var x;
      var y;
      var radius;
      var baseX;
      var baseY;
      var baseRadius;
      var scale;
      var size;

      if (!stageNode || !stage2Node) return;

      nodeRect = stageNode.getBoundingClientRect();
      x = nodeRect.left + nodeRect.width / 2 - stageRect.left;
      y = nodeRect.top + nodeRect.height / 2 - stageRect.top;
      radius = Math.max(16, Math.min(nodeRect.width, nodeRect.height) / 2);
      baseX = currentMeshGeometryOriginX + ((x - currentMeshGeometryOriginX) / transformScale);
      baseY = currentMeshGeometryOriginY + (((y - currentHeroVizStageY) - currentMeshGeometryOriginY) / transformScale);
      baseRadius = radius / transformScale;
      scale = Number.parseFloat(stage2Node.style.getPropertyValue('--stage2-node-scale')) || 1;
      size = nodeCardSize(id, scale);

      geometry[id] = { baseX: baseX, baseY: baseY, baseRadius: baseRadius, x: x, y: y, radius: radius, size: size, stage2Node: stage2Node };
      writeMeshNodeGeometry(id, geometry[id]);
    });

    cachedMeshGeometry = geometry;
    return cachedMeshGeometry;
  }

  function syncServerGeometry() {
    var scale;
    var stageRect;
    var nodeRect;
    var x;
    var y;
    var radius;

    if (!serverPlate || !sticky) return { x: window.innerWidth / 2, y: window.innerHeight / 2, radius: 34 };

    if (!cachedServer) {
      scale = currentMeshGeometryScale || 1;
      stageRect = sticky.getBoundingClientRect();
      nodeRect = serverPlate.getBoundingClientRect();
      x = nodeRect.left + nodeRect.width / 2 - stageRect.left;
      y = nodeRect.top + nodeRect.height / 2 - stageRect.top;
      radius = Math.max(18, Math.min(nodeRect.width, nodeRect.height) / 2);
      cachedServer = {
        baseX: x,
        baseY: y - currentHeroVizStageY,
        baseRadius: radius / scale,
        x: x,
        y: y,
        radius: radius,
      };
    }

    scale = currentMeshGeometryScale || 1;
    x = cachedServer.baseX;
    y = cachedServer.baseY + currentHeroVizStageY;
    radius = Math.max(18, cachedServer.baseRadius * scale);

    if (cachedServer.x !== x || cachedServer.y !== y) cachedApertureMaxRadius = null;
    cachedServer.x = x;
    cachedServer.y = y;
    cachedServer.radius = radius;

    setVar('--stage2-aperture-x', Math.round(x * 100) / 100 + 'px');
    setVar('--stage2-aperture-y', Math.round(y * 100) / 100 + 'px');
    setVar('--stage2-aperture-px', Math.round(x * 100) / 100 + 'px');
    setVar('--stage2-aperture-py', Math.round(y * 100) / 100 + 'px');
    setVar('--stage2-server-radius', Math.round(radius * 100) / 100 + 'px');

    return cachedServer;
  }

  function maxRadiusFrom(point) {
    if (cachedApertureMaxRadius !== null) return cachedApertureMaxRadius;

    var w = sticky ? sticky.clientWidth : window.innerWidth;
    var h = sticky ? sticky.clientHeight : window.innerHeight;
    var corners = [
      [0, 0],
      [w, 0],
      [0, h],
      [w, h],
    ];
    var max = 0;

    corners.forEach(function (corner) {
      var dx = corner[0] - point.x;
      var dy = corner[1] - point.y;
      max = Math.max(max, Math.sqrt(dx * dx + dy * dy));
    });

    cachedApertureMaxRadius = max + 80;
    return cachedApertureMaxRadius;
  }

  function linearStageProgressFromScrollProgress(progress) {
    return clamp((progress - STAGE_INTRO_PROGRESS) / (STAGE_ANIMATION_END_PROGRESS - STAGE_INTRO_PROGRESS), 0, 1) * SCRUBBER_END;
  }

  function linearScrollProgressFromStageProgress(progress) {
    return STAGE_INTRO_PROGRESS + (clamp(progress, 0, SCRUBBER_END) / SCRUBBER_END) * (STAGE_ANIMATION_END_PROGRESS - STAGE_INTRO_PROGRESS);
  }

  function useDesktopApertureEntryTiming() {
    return !useTouchScrollProfile();
  }

  function desktopApertureEntryScrollProgress() {
    return linearScrollProgressFromStageProgress(APERTURE_OPEN_START);
  }

  function desktopApertureEntryTargetScrollProgress() {
    return STAGE_INTRO_PROGRESS + (desktopApertureEntryScrollProgress() - STAGE_INTRO_PROGRESS) * DESKTOP_APERTURE_ENTRY_SCROLL_FACTOR;
  }

  function stageProgressFromScrollProgress(progress) {
    var clampedProgress = clamp(progress, 0, 1);
    var apertureEntryTarget;

    if (!useDesktopApertureEntryTiming()) return linearStageProgressFromScrollProgress(clampedProgress);
    if (clampedProgress <= STAGE_INTRO_PROGRESS) return 0;

    apertureEntryTarget = desktopApertureEntryTargetScrollProgress();
    if (clampedProgress <= apertureEntryTarget) {
      return lerp(0, APERTURE_OPEN_START, ramp(clampedProgress, STAGE_INTRO_PROGRESS, apertureEntryTarget));
    }

    return lerp(APERTURE_OPEN_START, SCRUBBER_END, ramp(clampedProgress, apertureEntryTarget, STAGE_ANIMATION_END_PROGRESS));
  }

  function scrollProgressFromStageProgress(progress) {
    var clampedProgress = clamp(progress, 0, SCRUBBER_END);
    var apertureEntryTarget;

    if (!useDesktopApertureEntryTiming()) return linearScrollProgressFromStageProgress(clampedProgress);
    if (clampedProgress <= 0) return STAGE_INTRO_PROGRESS;

    apertureEntryTarget = desktopApertureEntryTargetScrollProgress();
    if (clampedProgress <= APERTURE_OPEN_START) {
      return lerp(STAGE_INTRO_PROGRESS, apertureEntryTarget, ramp(clampedProgress, 0, APERTURE_OPEN_START));
    }

    return lerp(apertureEntryTarget, STAGE_ANIMATION_END_PROGRESS, ramp(clampedProgress, APERTURE_OPEN_START, SCRUBBER_END));
  }

  function modelShrinkProgressFromPlacement(stageProgress, phaseLocal) {
    var total = modelLayers.length || 10;
    var progress = 0;
    var index;
    var threshold;

    if (stageProgress < PHASES[SHRINK_PHASE_INDEX].start - 0.018) return 0;

    for (index = 0; index < total; index += 1) {
      threshold = 0.22 + index * 0.062;
      progress += ramp(phaseLocal, threshold - 0.035, threshold + 0.035);
    }

    return clamp(progress / total, 0, 1);
  }

  function setChromeProgress(progress, stageProgress) {
    var navOut = ramp(progress, 0.015, 0.18);
    var navReturn = stageProgress >= SCRUBBER_END ? 1 : 0;
    var navHidden = navOut * (1 - navReturn);
    var footerOut = ramp(stageProgress, FOOTER_CLEAR_START_STAGE_PROGRESS, FOOTER_CLEAR_END_STAGE_PROGRESS);
    var heroReady = document.body.classList.contains('is-hero-complete') || !document.body.classList.contains('home-intro');

    if (progress > 0.01) {
      document.body.classList.add('is-hero-complete');
      document.body.classList.remove('home-intro');
    }

    if (!heroReady && progress < 0.001) return;

    section.classList.toggle('is-stage2-footer-cleared', footerOut > 0.985);

    if (nav) {
      nav.style.opacity = String(1 - navHidden);
      nav.style.transform = 'translateY(' + Math.round(-88 * navHidden) + 'px)';
      nav.style.pointerEvents = navHidden > 0.92 ? 'none' : '';
    }

    syncTitleFromProgress(progress, stageProgress);

    if (heroFooter) {
      heroFooter.style.opacity = String(1 - footerOut);
      heroFooter.style.transform = 'translateY(' + Math.round(168 * easeOut(footerOut)) + 'px)';
      heroFooter.style.pointerEvents = footerOut > 0.8 ? 'none' : '';
    }
  }

  function applyProgress(progress) {
    var scrubberProgress;
    var shrinkPhaseProgress;
    var meshPhaseProgress;
    var alignPhaseProgress;
    var nodeRevealProgress;
    var nodeShapeProgress;
    var nodePositionProgress;
    var nodeContentOpacity;
    var nodeIconOpacity;
    var modelShrinkProgress;
    var serverReturnProgress;
    var apertureMaxRadius;
    var apertureScale;

    progress = clamp(progress, 0, SCRUBBER_END);
    var stageProgress = stageProgressFromScrollProgress(progress);
    var server;
    var apertureOpen = ramp(stageProgress, APERTURE_OPEN_START, APERTURE_OPEN_END);
    var apertureCloseRaw = ramp(stageProgress, APERTURE_CLOSE_START, APERTURE_CLOSE_END);
    var apertureClose = fastCloseEase(apertureCloseRaw);
    var aperture = clamp(apertureOpen - apertureClose, 0, 1);
    var apertureFadeIn = easeOut(ramp(stageProgress, APERTURE_OPEN_START, 0.292));
    var apertureFadeOut = easeInOut(ramp(stageProgress, APERTURE_FADE_START, APERTURE_FADE_END));
    var light = clamp(apertureFadeIn - apertureFadeOut, 0, 1);
    var closeDetailFade = easeOut(ramp(stageProgress, APERTURE_CLOSE_START, APERTURE_DETAIL_FADE_END));
    var workbench = clamp(ramp(stageProgress, 0.360, 0.435) - easeInOut(ramp(stageProgress, APERTURE_CLOSE_START, APERTURE_WORKBENCH_EXIT_END)), 0, 1);
    var heroGridRetreat = easeInOut(ramp(stageProgress, PHASES[LOCK_PHASE_INDEX].end, APERTURE_CLOSE_END));
    var heroGrid = 1 - (heroGridRetreat * 0.46);
    var apertureRadius;
    var ringOpacity = clamp(ramp(stageProgress, 0.235, 0.270) - ramp(stageProgress, 0.365, 0.430) + ramp(stageProgress, APERTURE_CLOSE_START, APERTURE_RING_RETURN_END) - ramp(stageProgress, APERTURE_FADE_START, APERTURE_CLOSE_END), 0, 1);
    var meshZoom = clamp(ramp(stageProgress, 0.165, APERTURE_OPEN_END) - easeInOut(ramp(stageProgress, APERTURE_CLOSE_START, APERTURE_CLOSE_END)), 0, 1);
    var meshScale = lerp(1, 1.115, easeInOut(meshZoom));
    var scrubberEnter = ramp(stageProgress, 0.360, 0.435);
    var scrubberExit = easeInOut(ramp(stageProgress, APERTURE_CLOSE_START, APERTURE_SCRUBBER_EXIT_END));
    var scrubberPresence = clamp(scrubberEnter - scrubberExit, 0, 1);
    var scrubberEnterY = (1 - scrubberEnter) * 16;
    var scrubberExitY = scrubberExit * 112;
    shrinkPhaseProgress = phaseProgress(PHASES[SHRINK_PHASE_INDEX], stageProgress);
    meshPhaseProgress = phaseProgress(PHASES[NODE_PHASE_INDEX], stageProgress);
    alignPhaseProgress = phaseProgress(PHASES[ALIGN_PHASE_INDEX], stageProgress);
    modelShrinkProgress = modelShrinkProgressFromPlacement(stageProgress, shrinkPhaseProgress);
    serverReturnProgress = easeOutQuart(modelShrinkProgress);
    nodeRevealProgress = easeOutExpo(ramp(meshPhaseProgress, 0.04, 0.30));
    nodeShapeProgress = easeInOut(ramp(meshPhaseProgress, 0.30, 0.58)) * (1 - easeInOut(ramp(alignPhaseProgress, 0.28, 0.56)));
    nodePositionProgress = easeInOut(ramp(meshPhaseProgress, 0.40, 0.66)) * (1 - easeInOut(ramp(alignPhaseProgress, 0.58, 0.92)));
    nodeContentOpacity = nodeShapeProgress * easeOutQuart(ramp(meshPhaseProgress, 0.52, 0.70)) * (1 - easeOutQuart(ramp(alignPhaseProgress, 0.04, 0.24)));
    nodeIconOpacity = nodeRevealProgress * (1 - nodeShapeProgress) * (1 - closeDetailFade);

    setStage2TitleProgress(stageProgress);
    setVar('--stage2-p', stageProgress);
    setVar('--stage2-pct', Math.round((stageProgress / SCRUBBER_END) * 10000) / 100 + '%');
    setVar('--stage2-scrubber-pct', Math.round(scrubberProgressFromProgress(stageProgress) * 10000) / 100 + '%');
    setVar('--stage2-aperture-ring-opacity', ringOpacity);
    setVar('--stage2-light-opacity', light);
    setVar('--stage2-workbench-opacity', workbench);
    setVar('--stage2-aperture-detail-fade', closeDetailFade);
    setVar('--stage2-node-card-progress', nodeShapeProgress);
    setVar('--stage2-node-shape-progress', nodeShapeProgress);
    setVar('--stage2-node-position-progress', nodePositionProgress);
    setVar('--stage2-node-content-opacity', nodeContentOpacity);
    setVar('--stage2-node-icon-opacity', nodeIconOpacity);
    setVar('--stage2-model-shrink-progress', modelShrinkProgress);
    setVar('--stage2-server-return-progress', serverReturnProgress);
    setVar('--stage2-scrubber-y', Math.round((scrubberEnterY + scrubberExitY) * 100) / 100 + 'px');
    setVar('--stage2-scrubber-opacity', scrubberPresence);
    syncScrubberAccessibility(scrubberPresence);
    syncPlacedLayerAccessibility(nodeShapeProgress);
    setVar('--stage2-grid-opacity', heroGrid);
    setVar('--stage2-grid-mask-solid', Math.round(lerp(78, 28, heroGridRetreat) * 100) / 100 + '%');
    setVar('--stage2-grid-mask-end', Math.round(lerp(100, 38, heroGridRetreat) * 100) / 100 + '%');
    setVar('--stage2-aperture-blue-alpha', lerp(0.08, 0.014, closeDetailFade));
    setVar('--stage2-aperture-grid-alpha', lerp(0.045, 0.006, closeDetailFade));
    if (!heroVizStableRequested && stageProgress >= APERTURE_CLOSE_END) {
      heroVizStableRequested = true;
      window.dispatchEvent(new CustomEvent('mesh:hero-viz:stable'));
    }
    section.classList.toggle('is-stage2-scrubbable', scrubberPresence > 0.12);

    setChromeProgress(progress, stageProgress);
    applyHeroVizTransform(meshScale);
    server = syncServerGeometry();
    apertureMaxRadius = maxRadiusFrom(server);
    apertureRadius = lerp(server.radius * 1.14, apertureMaxRadius, easeInOut(aperture));
    apertureScale = apertureMaxRadius > 0 ? clamp(apertureRadius / apertureMaxRadius, 0, 1) : 0;
    setVar('--stage2-aperture-radius', Math.round(apertureRadius * 100) / 100 + 'px');
    setVar('--stage2-aperture-max-radius', Math.round(apertureMaxRadius * 100) / 100 + 'px');
    setVar('--stage2-aperture-scale', Math.round(apertureScale * 10000) / 10000);
    setVar('--stage2-aperture-inverse-scale', apertureScale > 0.001 ? Math.round((1 / apertureScale) * 10000) / 10000 : 1);

    syncMeshGeometry();
    clearMeshNodeLayoutTransforms();
    clearIdleSourceLayerTransforms();

    syncPhaseState(stageProgress);
    applyPhaseScaffold(stageProgress);
    syncLayerCompaction(stageProgress);
    ensureAutoPlacement(stageProgress);
    scheduleIdleSourceLayerTransformClear();

    scrubberProgress = scrubberProgressFromProgress(stageProgress);
    if (scrubInput && Math.abs(Number(scrubInput.value) / 1000 - scrubberProgress) > 0.003) {
      scrubInput.value = Math.round(scrubberProgress * 1000);
    }

    if (!reduceMotion) {
      document.body.classList.toggle('is-stage2-active', stageProgress > 0.002 && stageProgress < SCRUBBER_END);
      document.body.classList.toggle('is-stage2-aperture-active', stageProgress >= APERTURE_CLOSE_START && stageProgress < SCRUBBER_END);
      document.body.classList.toggle('is-stage2-complete', stageProgress >= SCRUBBER_END);
    } else {
      document.body.classList.remove('is-stage2-aperture-active');
    }

    if (stageProgress >= SCRUBBER_END - 0.01) syncCompletedGridClip();
  }

  function progressFromScroll() {
    var start = section.offsetTop;
    var range = Math.max(1, section.offsetHeight - window.innerHeight);
    return clamp((window.scrollY - start) / range, 0, 1);
  }

  function scrollToProgress(progress) {
    var start = section.offsetTop;
    var range = Math.max(1, section.offsetHeight - window.innerHeight);
    window.scrollTo({ top: start + range * scrollProgressFromStageProgress(progress), behavior: 'auto' });
  }

  function scrollTopForScrollProgress(progress) {
    var start = section.offsetTop;
    var range = Math.max(1, section.offsetHeight - window.innerHeight);
    return start + range * clamp(progress, 0, 1);
  }

  function scrollTopForStageProgress(progress) {
    var start = section.offsetTop;
    var range = Math.max(1, section.offsetHeight - window.innerHeight);
    return start + range * scrollProgressFromStageProgress(progress);
  }

  function forceInstantScrollBehavior() {
    if (scrubScrollBehaviorRestore !== null) return;
    scrubScrollBehaviorRestore = document.documentElement.style.scrollBehavior;
    document.documentElement.style.scrollBehavior = 'auto';
  }

  function restoreScrollBehavior() {
    if (scrubScrollBehaviorRestore === null) return;
    document.documentElement.style.scrollBehavior = scrubScrollBehaviorRestore;
    scrubScrollBehaviorRestore = null;
  }

  function cancelScrubScrollAnimation() {
    scrubScrollToken += 1;
    if (scrubScrollRaf) window.cancelAnimationFrame(scrubScrollRaf);
    scrubScrollRaf = null;
    restoreScrollBehavior();
    if (window.anime && window.anime.remove && scrubScrollState) window.anime.remove(scrubScrollState);
    scrubScrollState = null;
  }

  function cancelScrollSmoothing() {
    if (scrollSmoothRaf) window.cancelAnimationFrame(scrollSmoothRaf);
    scrollSmoothRaf = null;
  }

  function beginManualStageSeekOverride() {
    manualStageSeekActive = true;
    manualStageSeekScrollY = window.scrollY;
  }

  function clearManualStageSeekOverride() {
    manualStageSeekActive = false;
  }

  function hasManualStageSeekOverride() {
    if (!manualStageSeekActive) return false;

    if (!isScrubbing && Math.abs(window.scrollY - manualStageSeekScrollY) > 2) {
      clearManualStageSeekOverride();
      return false;
    }

    return true;
  }

  function clearManualStageSeekOverrideOnUserIntent(event) {
    var target;

    if (!manualStageSeekActive) return;

    target = event && event.target;
    if (target && target.closest && target.closest('.stage2-scrubber')) return;

    clearManualStageSeekOverride();
  }

  function scrubToStageProgress(progress, onComplete, options) {
    var token;
    var from;
    var fromStage;
    var to;
    var distance;
    var duration;
    var referenceDistance;
    var state;
    var startTime;

    function completeScrub() {
      if (token !== scrubScrollToken) return;
      window.scrollTo({ top: to, behavior: 'auto' });
      smoothTo(scrollProgressFromStageProgress(progress), true);
      scrubScrollState = null;
      scrubScrollRaf = null;
      restoreScrollBehavior();
      if (typeof onComplete === 'function') onComplete();
    }

    function stepScrub(now) {
      var elapsed;
      var local;
      var eased;
      var top;

      if (token !== scrubScrollToken) return;

      elapsed = Math.max(0, now - startTime);
      local = clamp(elapsed / duration, 0, 1);
      eased = local;
      top = lerp(from, to, eased);

      window.scrollTo({ top: top, behavior: 'auto' });
      smoothTo(progressFromScroll(), true);

      if (local >= 1) {
        completeScrub();
        return;
      }

      scrubScrollRaf = window.requestAnimationFrame(stepScrub);
    }

    progress = clamp(progress, 0, SCRUBBER_END);

    clearManualStageSeekOverride();
    if (window.anime && window.anime.remove) window.anime.remove(smoothState);
    cancelScrubScrollAnimation();
    token = scrubScrollToken;

    from = window.scrollY;
    fromStage = Number.parseFloat(section.style.getPropertyValue('--stage2-p')) || stageProgressFromScrollProgress(progressFromScroll());
    to = scrollTopForStageProgress(progress);
    distance = Math.abs(to - from);
    referenceDistance = options && options.referenceDistance ? Math.max(1, Math.abs(options.referenceDistance)) : 0;
    duration = clamp(
      Math.abs(progress - fromStage) * 16000,
      720,
      Math.min(2200, Math.max(900, distance * 0.85))
    ) * LEARN_MORE_SCRUB_SPEED_MULTIPLIER;
    if (options && options.preserveScrollSpeed && referenceDistance) {
      duration *= distance / referenceDistance;
    }
    state = { top: from, stage: fromStage };
    scrubScrollState = state;
    forceInstantScrollBehavior();

    if (reduceMotion) {
      window.scrollTo({ top: to, behavior: 'auto' });
      smoothTo(scrollProgressFromStageProgress(progress), true);
      scrubScrollState = null;
      restoreScrollBehavior();
      if (token === scrubScrollToken && typeof onComplete === 'function') onComplete();
      return;
    }

    startTime = performance.now();
    scrubScrollRaf = window.requestAnimationFrame(stepScrub);
  }

  function seekStageProgressImmediate(progress) {
    var scrollProgress = scrollProgressFromStageProgress(progress);

    cancelScrollSmoothing();
    if (window.anime && window.anime.remove) window.anime.remove(smoothState);
    cancelScrubScrollAnimation();

    latestScrollProgress = scrollProgress;
    smoothState.progress = scrollProgress;
    timelineState.progress = scrollProgress;
    beginManualStageSeekOverride();
    applyProgress(scrollProgress);
  }

  function seek(progress) {
    progress = clamp(progress, 0, 1);

    if (timeline && timeline.seek) {
      try {
        timeline.seek(progress * 1000);
      } catch (_error) {
        applyProgress(progress);
      }
    } else {
      applyProgress(progress);
    }
  }

  function smoothTo(progress, immediate) {
    var delta;
    var snap;

    progress = clamp(progress, 0, 1);
    latestScrollProgress = progress;

    if (immediate || reduceMotion || useTouchScrollProfile() || !window.anime || !window.anime.animate) {
      cancelScrollSmoothing();
      if (window.anime && window.anime.remove) window.anime.remove(smoothState);
      smoothState.progress = progress;
      seek(progress);
      return;
    }

    if (window.anime.remove) window.anime.remove(smoothState);

    if (scrollSmoothRaf) return;

    function renderSmoothedProgress() {
      scrollSmoothRaf = null;
      delta = latestScrollProgress - smoothState.progress;

      if (Math.abs(delta) < 0.0006) {
        smoothState.progress = latestScrollProgress;
        seek(smoothState.progress);
        return;
      }

      snap = smoothState.progress < latestScrollProgress && latestScrollProgress === 1 ? 0.0001 : 0;
      snap = smoothState.progress > latestScrollProgress && latestScrollProgress === 0 ? -0.0001 : snap;
      smoothState.progress = clamp(smoothState.progress + delta * scrollSmoothStep() + snap, 0, 1);
      seek(smoothState.progress);
      scrollSmoothRaf = window.requestAnimationFrame(renderSmoothedProgress);
    }

    scrollSmoothRaf = window.requestAnimationFrame(renderSmoothedProgress);
  }

  function requestUpdate(immediate) {
    if (raf) return;
    raf = window.requestAnimationFrame(function () {
      raf = null;
      if (hasManualStageSeekOverride()) return;
      smoothTo(progressFromScroll(), immediate || isScrubbing);
    });
  }

  function handleGeometryChange() {
    syncHeroChromeMetrics();
    invalidateGeometry();
    syncCompletedGridClip();
    requestUpdate(true);
  }

  function initStage2Layout() {
    if (!layoutRoot || !window.anime || typeof window.anime.createLayout !== 'function') return;

    try {
      stage2Layout = window.anime.createLayout(layoutRoot, {
        duration: 720,
        ease: 'inOutExpo',
      });
    } catch (_error) {
      stage2Layout = null;
    }
  }

  function cancelSliceCutterRetarget() {
    if (sliceCutterRaf) window.cancelAnimationFrame(sliceCutterRaf);
    sliceCutterRaf = null;
  }

  function retargetSliceCutter(cutIndex) {
    var stackRect;
    var boundaryLayer;
    var boundaryRect;
    var cutterY;

    if (!tensorStack || !modelLayers.length) return false;

    stackRect = tensorStack.getBoundingClientRect();
    boundaryLayer = modelLayers[Math.min(Math.max(cutIndex, 0), modelLayers.length - 1)];
    boundaryRect = boundaryLayer && boundaryLayer.getBoundingClientRect();

    if (!stackRect || !boundaryRect) return false;

    cutterY = Math.max(0, boundaryRect.bottom - stackRect.top);
    setVar('--stage2-cutter-y', Math.round(cutterY * 100) / 100 + 'px');
    return true;
  }

  function scheduleSliceCutterRetarget(cutIndex) {
    cancelSliceCutterRetarget();
    sliceCutterRaf = window.requestAnimationFrame(function () {
      sliceCutterRaf = null;
      retargetSliceCutter(cutIndex);
    });
  }

  function paintSliceTimeline(progress) {
    var cumulativeGap = 0;
    var cutCount = Math.max(1, modelLayers.length || 10);
    var cutIndex = Math.min(cutCount - 1, Math.floor(clamp(progress, 0, 0.999) * cutCount));
    var cutLocal = clamp(progress * cutCount - cutIndex, 0, 1);
    var cutterY = ((cutIndex + 1) / cutCount) * 100;
    var gapSize = 8;
    var travelProgress = easeOutQuart(ramp(cutLocal, 0.04, 0.48));
    var followProgress = easeOutQuart(ramp(cutLocal, 0.48, 0.72));
    var activeGapProgress = easeOutQuart(ramp(cutLocal, 0.72, 1));
    var cutterLeft = followProgress;
    var cutterRight = followProgress > 0 ? 1 : travelProgress;
    var cutterVisible = progress > 0.002 && progress < 0.998 && cutLocal < 0.72;
    var cutterEnergy = cutterVisible ? easeOutQuart(ramp(cutLocal, 0.02, 0.22)) * (1 - easeOutQuart(ramp(cutLocal, 0.56, 0.78))) : 0;

    progress = clamp(progress, 0, 1);
    setVar('--stage2-slice-progress', progress);
    setVar('--stage2-cutter-left', Math.round(cutterLeft * 10000) / 10000);
    setVar('--stage2-cutter-right', Math.round(cutterRight * 10000) / 10000);
    setVar('--stage2-cutter-opacity', cutterVisible ? 1 : 0);
    setVar('--stage2-cutter-energy', Math.round(cutterEnergy * 10000) / 10000);

    modelLayers.forEach(function (layer, index) {
      var release = index < cutIndex ? 1 : index === cutIndex ? activeGapProgress : 0;
      var previousRelease = index > 0 ? index - 1 < cutIndex ? 1 : index - 1 === cutIndex ? activeGapProgress : 0 : 0;
      var layerGap = previousRelease * gapSize;
      var layerLocal = progress * cutCount - index;
      var impact = easeOutQuart(ramp(layerLocal, 0.02, 0.34)) * (1 - easeOutQuart(ramp(layerLocal, 0.42, 0.92)));
      var residue = easeOutQuart(ramp(layerLocal, 0.34, 0.68)) * (1 - easeOutQuart(ramp(layerLocal, 0.88, 1.34)));
      var cutFront = easeOutQuart(ramp(layerLocal, 0.02, 0.58));
      var shear = useLiteStage2Motion() ? 0 : impact * (index % 2 === 0 ? -1 : 1) * 1.8;

      setElementVar(layer, '--stage2-layer-slice-progress', String(Math.round(release * 10000) / 10000));
      setElementVar(layer, '--stage2-layer-gap-before', Math.round(layerGap * 100) / 100 + 'px');
      setElementVar(layer, '--stage2-layer-cut-energy', String(Math.round(impact * 10000) / 10000));
      setElementVar(layer, '--stage2-layer-cut-residue', String(Math.round(residue * 10000) / 10000));
      setElementVar(layer, '--stage2-layer-cut-front', String(Math.round(cutFront * 10000) / 10000));
      setElementVar(layer, '--stage2-layer-cut-shear', Math.round(shear * 100) / 100 + 'px');
      setElementVar(layer, '--stage2-layer-cut-skew', Math.round(shear * -0.32 * 100) / 100 + 'deg');
      if (layer.parentElement === tensorStack) cumulativeGap += layerGap;
    });

    setVar('--stage2-stack-extra-height', Math.round(cumulativeGap * 100) / 100 + 'px');

    if (useLiteStage2Motion()) {
      cancelSliceCutterRetarget();
      setVar('--stage2-cutter-y', cutterY + '%');
      return;
    }

    if (retargetSliceCutter(cutIndex)) {
      if (!isScrubbing) scheduleSliceCutterRetarget(cutIndex);
      return;
    }

    setVar('--stage2-cutter-y', cutterY + '%');
  }

  function initSliceTimeline() {
    if (!window.anime || typeof window.anime.createTimeline !== 'function') return;

    try {
      sliceTimeline = window.anime.createTimeline({
        autoplay: false,
        defaults: { ease: 'linear' },
      });

      sliceTimeline.add(sliceTimelineState, {
        progress: [0, 1],
        duration: 1000,
        ease: 'linear',
        onUpdate: function () {
          sliceTimelinePainted = true;
          paintSliceTimeline(sliceTimelineState.progress);
        },
      }, 0);
    } catch (_error) {
      sliceTimeline = null;
    }
  }

  function syncSliceTimeline(progress) {
    progress = clamp(progress, 0, 1);
    sliceTimelineState.progress = progress;

    if (sliceTimeline && typeof sliceTimeline.seek === 'function') {
      try {
        sliceTimeline.seek(progress * 1000, true);
        paintSliceTimeline(progress);
      } catch (_error) {
        paintSliceTimeline(progress);
      }
      return;
    }

    paintSliceTimeline(progress);
  }

  function syncLayerCompaction(stageProgress) {
    var phaseLocal = phaseProgress(PHASES[SHRINK_PHASE_INDEX], stageProgress);
    var placementPhaseStarted = stageProgress >= PHASES[MESH_PHASE_INDEX].start;

    modelLayers.forEach(function (layer, index) {
      var compact = phaseLocal > 0 ? easeOutQuart(ramp(phaseLocal, 0.04 + index * 0.035, 0.18 + index * 0.035)) : 0;
      if (layer.getAttribute('data-stage2-assigned-node')) compact = 1;
      else if (placementPhaseStarted && !layer.classList.contains('is-stage2-layer-in-flight')) compact = 0;
      setElementVar(layer, '--stage2-layer-compact-progress', String(Math.round(compact * 10000) / 10000));
    });
  }

  function svgElement(name, attrs, parent) {
    var node = document.createElementNS('http://www.w3.org/2000/svg', name);
    Object.keys(attrs || {}).forEach(function (key) {
      node.setAttribute(key, attrs[key]);
    });
    if (parent) parent.appendChild(node);
    return node;
  }

  function renderStage2NodeIcons() {
    if (!window.lucide || !window.lucide.icons) return;

    meshNodes.forEach(function (node) {
      var iconHost = node.querySelector('[data-stage2-node-icon]');
      var iconName = iconHost && iconHost.getAttribute('data-stage2-node-icon');
      var icon = iconName && window.lucide.icons[iconName];
      var svg;
      var inner;

      if (!iconHost || !icon || iconHost.querySelector('svg')) return;

      svg = svgElement('svg', {
        viewBox: '0 0 24 24',
        fill: 'none',
        stroke: 'currentColor',
        'stroke-width': '1.9',
        'stroke-linecap': 'round',
        'stroke-linejoin': 'round',
        focusable: 'false',
        'aria-hidden': 'true',
      }, iconHost);
      inner = svgElement('g', {}, svg);
      icon.forEach(function (shape) {
        svgElement(shape[0], shape[1] || {}, inner);
      });
    });
  }

  function clearMeshNodeLayoutTransforms() {
    meshNodes.forEach(function (node) {
      node.style.transform = '';
      node.style.translate = '';
    });
  }

  function cancelLayoutClearTimers() {
    layoutClearTimers.forEach(function (timerId) {
      window.clearTimeout(timerId);
    });
    layoutClearTimers = [];
  }

  function scheduleLayoutClear(callback, delay) {
    var timerId = window.setTimeout(function () {
      layoutClearTimers = layoutClearTimers.filter(function (candidate) {
        return candidate !== timerId;
      });
      callback();
    }, Math.max(0, delay || 0));

    layoutClearTimers.push(timerId);
  }

  function hasActivePlacementSourceMotion() {
    return modelLayers.some(function (layer) {
      var state = layer.__stage2PlacementData;
      return Boolean(state && state.reservation && state.reservation.parentElement);
    });
  }

  function clearIdleSourceLayerTransforms() {
    if (hasActivePlacementSourceMotion()) return;

    modelLayers.forEach(function (layer) {
      if (layer.getAttribute('data-stage2-assigned-node')) return;
      layer.style.transform = '';
      layer.style.translate = '';
    });
  }

  function scheduleIdleSourceLayerTransformClear() {
    clearIdleSourceLayerTransforms();
    scheduleLayoutClear(clearIdleSourceLayerTransforms, 40);
  }

  function clearSettledLayerTransforms() {
    modelLayers.forEach(function (layer) {
      layer.style.transform = '';
      layer.style.translate = '';
    });
  }

  function scheduleSettledLayerTransformClear(duration) {
    var delay = Number(duration) || 0;
    scheduleLayoutClear(clearSettledLayerTransforms, delay + 50);
  }

  function scheduleMeshNodeTransformClear(duration) {
    clearMeshNodeLayoutTransforms();
    scheduleLayoutClear(clearMeshNodeLayoutTransforms, Math.max(0, duration || 0) + 40);
  }

  function runLayoutMutation(mutation, options) {
    var layoutOptions = options || { duration: 640, ease: 'inOutExpo' };
    var duration = Number(layoutOptions.duration) || 0;

    cancelLayoutClearTimers();

    if (stage2Layout && typeof stage2Layout.update === 'function') {
      try {
        stage2Layout.update(mutation, layoutOptions);
        scheduleMeshNodeTransformClear(duration);
        scheduleSettledLayerTransformClear(duration);
        return;
      } catch (_error) {
        // Fall through to the explicit record/animate pattern documented by Anime Layout.
      }
    }

    if (stage2Layout && typeof stage2Layout.record === 'function' && typeof stage2Layout.animate === 'function') {
      try {
        stage2Layout.record();
        mutation();
        stage2Layout.animate(layoutOptions);
        scheduleMeshNodeTransformClear(duration);
        scheduleSettledLayerTransformClear(duration);
        return;
      } catch (_error) {
        // Fall through to direct DOM mutation so the story stays truthful even if Layout is unavailable.
      }
    }

    mutation();
    clearMeshNodeLayoutTransforms();
    scheduleIdleSourceLayerTransformClear();
    scheduleSettledLayerTransformClear(0);
  }

  function slotForNode(nodeId) {
    return section.querySelector('[data-stage2-node-slots="' + nodeId + '"]');
  }

  function colorForNode(nodeId) {
    var node = nodeId && section.querySelector('[data-stage2-node="' + nodeId + '"]');
    var color;

    if (!node) return '';
    color = window.getComputedStyle(node).getPropertyValue('--stage2-node-color').trim();
    return color || '';
  }

  function layerIdFor(layer) {
    return layer ? layer.getAttribute('data-stage2-model-layer') : '';
  }

  function setLayerSourceSemantics(layer) {
    layer.removeAttribute('aria-disabled');
    layer.removeAttribute('aria-label');
    layer.removeAttribute('title');
    layer.removeAttribute('tabindex');
    layer.setAttribute('data-stage2-layer-state', 'source');
  }

  function setLayerPlacedSemantics(layer, nodeId) {
    layer.removeAttribute('aria-disabled');
    layer.removeAttribute('aria-label');
    layer.removeAttribute('title');
    layer.removeAttribute('tabindex');
    layer.setAttribute('data-stage2-layer-state', 'placed');
  }

  function setLayerReducedSemantics(layer, nodeId) {
    layer.removeAttribute('aria-disabled');
    layer.removeAttribute('aria-label');
    layer.removeAttribute('title');
    layer.removeAttribute('tabindex');
    layer.setAttribute('data-stage2-layer-state', 'summary');
  }

  function setLayerCollapsedSemantics(layer, nodeId) {
    layer.removeAttribute('aria-disabled');
    layer.removeAttribute('aria-label');
    layer.removeAttribute('title');
    layer.removeAttribute('tabindex');
    layer.setAttribute('data-stage2-layer-state', 'collapsed');
  }

  function syncPlacedLayerAccessibility(nodeCardProgress) {
    if (!modelLayers.length || reduceMotion) return;

    modelLayers.forEach(function (layer) {
      var nodeId = layer.getAttribute('data-stage2-assigned-node');
      if (!nodeId) return;

      if (nodeCardProgress > 0.36) {
        setLayerPlacedSemantics(layer, nodeId);
        return;
      }

      setLayerCollapsedSemantics(layer, nodeId);
    });
  }

  function syncScrubberAccessibility(scrubberPresence) {
    var scrubber = section.querySelector('.stage2-scrubber');
    var isVisible = scrubberPresence > 0.12;

    if (scrubber) scrubber.setAttribute('aria-hidden', isVisible ? 'false' : 'true');
    if (!scrubInput) return;

    scrubInput.disabled = !isVisible;
    scrubInput.tabIndex = isVisible ? 0 : -1;
  }

  function insertLayerInTensorOrder(layer) {
    var layerIndex = Number(layerIdFor(layer));
    var cutter = tensorStack && tensorStack.querySelector('[data-stage2-slice-cutter]');
    var before = null;

    if (!tensorStack || !layer) return;

    modelLayers.some(function (candidate) {
      if (candidate === layer || candidate.parentElement !== tensorStack) return false;
      if (Number(layerIdFor(candidate)) <= layerIndex) return false;
      before = candidate;
      return true;
    });

    tensorStack.insertBefore(layer, before || cutter || null);
  }

  function insertLayerInNodeOrder(layer, targetParent) {
    var layerIndex = Number(layerIdFor(layer));
    var before = null;

    if (!layer || !targetParent) return;

    modelLayers.some(function (candidate) {
      if (candidate === layer || candidate.parentElement !== targetParent) return false;
      if (Number(layerIdFor(candidate)) <= layerIndex) return false;
      before = candidate;
      return true;
    });

    targetParent.insertBefore(layer, before || null);
  }

  function rebuildLayerAssignments() {
    layerAssignments = {};
    modelLayers.forEach(function (layer) {
      var nodeId = layer.getAttribute('data-stage2-assigned-node');
      if (nodeId) layerAssignments[layerIdFor(layer)] = nodeId;
    });

    placementStarted = modelLayers.some(function (layer) {
      return Boolean(layer.getAttribute('data-stage2-assigned-node'));
    });
    autoPlacementComplete = modelLayers.length > 0 && modelLayers.every(function (layer) {
      return Boolean(layer.getAttribute('data-stage2-assigned-node'));
    });

    section.classList.toggle('is-stage2-placement-started', placementStarted);
    if (!placementStarted) clearVar('--stage2-source-layer-height');
  }

  function ensureStage2FlightLayer() {
    if (stage2FlightLayer && stage2FlightLayer.isConnected) return stage2FlightLayer;
    if (!layoutRoot) return null;

    stage2FlightLayer = layoutRoot.querySelector('[data-stage2-flight-layer]');
    if (stage2FlightLayer) return stage2FlightLayer;

    stage2FlightLayer = document.createElement('div');
    stage2FlightLayer.className = 'stage2-flight-layer';
    stage2FlightLayer.setAttribute('data-stage2-flight-layer', 'true');
    stage2FlightLayer.setAttribute('aria-hidden', 'true');
    layoutRoot.appendChild(stage2FlightLayer);
    return stage2FlightLayer;
  }

  function clearLayerFlightStyles(layer) {
    layer.classList.remove('is-stage2-layer-in-flight');
    layer.style.position = '';
    layer.style.left = '';
    layer.style.top = '';
    layer.style.width = '';
    layer.style.height = '';
    layer.style.marginTop = '';
    layer.style.transform = '';
    layer.style.translate = '';
    layer.style.willChange = '';
    clearElementVar(layer, '--stage2-flight-shadow-y');
    clearElementVar(layer, '--stage2-flight-shadow-blur');
    clearElementVar(layer, '--stage2-flight-shadow-spread');
    clearElementVar(layer, '--stage2-flight-shadow-alpha');
    clearElementVar(layer, '--stage2-flight-lift-y');
    clearElementVar(layer, '--stage2-flight-scale');
  }

  function clearLayerPlacementStyles(layer) {
    if (!layer) return;

    clearLayerFlightStyles(layer);
    layer.__stage2MoveState = null;
    layer.__stage2MoveTargetParent = null;
    clearElementVar(layer, '--stage2-layer-placed-opacity');
    clearElementVar(layer, '--stage2-layer-node-color');
  }

  function setLayerFlightVisuals(layer, progress) {
    var crisp = easeInOut(progress);
    var lift = 1 - easeOutQuart(progress);

    if (!layer) return;

    setElementVar(layer, '--stage2-flight-shadow-y', Math.round(lerp(24, 2, crisp) * 100) / 100 + 'px');
    setElementVar(layer, '--stage2-flight-shadow-blur', Math.round(lerp(30, 3, crisp) * 100) / 100 + 'px');
    setElementVar(layer, '--stage2-flight-shadow-spread', Math.round(lerp(-5, 0, crisp) * 100) / 100 + 'px');
    setElementVar(layer, '--stage2-flight-shadow-alpha', String(Math.round(lerp(0.24, 0.08, crisp) * 10000) / 10000));
    setElementVar(layer, '--stage2-flight-lift-y', Math.round(lerp(-5, 0, crisp) * lift * 100) / 100 + 'px');
    setElementVar(layer, '--stage2-flight-scale', String(Math.round(lerp(1.012, 1, crisp) * 10000) / 10000));
  }

  function hasActiveLayerFlight() {
    return modelLayers.some(function (layer) {
      return layer.parentElement === stage2FlightLayer || Boolean(layer.__stage2MoveState);
    });
  }

  function isPlacementSettled() {
    return modelLayers.length > 0 && modelLayers.every(function (layer) {
      return Boolean(layer.getAttribute('data-stage2-assigned-node'));
    }) && !placementMotionActive && !hasActiveLayerFlight();
  }

  function settleAllPlacementTargets() {
    var changed = false;

    modelLayers.forEach(function (layer) {
      var layerId = layerIdFor(layer);
      var targetNode = DEFAULT_ASSIGNMENTS[layerId] || null;
      var targetParent = targetNode ? slotForNode(targetNode) : null;
      var state = layer.__stage2PlacementData;

      if (!targetNode || !targetParent) return;

      if (state) {
        removePlacementReservation(state);
        layer.__stage2PlacementData = null;
      }

      if (layer.getAttribute('data-stage2-assigned-node') !== targetNode || layer.parentElement !== targetParent) {
        settleLayerPlacementState(layer, targetParent, targetNode);
        changed = true;
      }

      if (layer.classList.contains('is-stage2-layer-in-flight') || layer.style.position === 'absolute') {
        clearLayerFlightStyles(layer);
        changed = true;
      }
    });

    if (!changed) return false;
    clearSourceMagazineMotion();
    placementMotionActive = false;
    placementSettleBarrier = false;
    rebuildLayerAssignments();
    return true;
  }

  function syncTouchStaticPlacement(stageProgress) {
    var phaseLocal = phaseProgress(PHASES[MESH_PHASE_INDEX], stageProgress);
    var shouldPlace = stageProgress >= PHASES[LOCK_PHASE_INDEX].start || phaseLocal >= TOUCH_PLACEMENT_SNAP_PROGRESS;
    var changed = false;

    if (shouldPlace && autoPlacementComplete && !placementMotionActive && !hasActiveLayerFlight()) return;
    if (!shouldPlace && !autoPlacementComplete && !placementStarted && !hasActiveLayerFlight()) return;

    cancelLayoutClearTimers();

    modelLayers.forEach(function (layer) {
      var layerId = layerIdFor(layer);
      var targetNode = shouldPlace ? DEFAULT_ASSIGNMENTS[layerId] || null : null;
      var targetParent = targetNode ? slotForNode(targetNode) : tensorStack;
      var state = layer.__stage2PlacementData;

      if (!targetParent) return;

      if (window.anime && window.anime.remove && layer.__stage2MoveState) {
        window.anime.remove(layer.__stage2MoveState);
      }

      if (state) {
        removePlacementReservation(state);
        layer.__stage2PlacementData = null;
      }

      if (layer.parentElement === stage2FlightLayer || layer.classList.contains('is-stage2-layer-in-flight') || layer.style.position === 'absolute') {
        clearLayerFlightStyles(layer);
        changed = true;
      }

      if (targetNode) {
        if (layer.getAttribute('data-stage2-assigned-node') !== targetNode || layer.parentElement !== targetParent) {
          settleLayerPlacementState(layer, targetParent, targetNode);
          changed = true;
        }
      } else if (layer.getAttribute('data-stage2-assigned-node') || layer.parentElement !== tensorStack) {
        settleLayerPlacementState(layer, tensorStack, null);
        changed = true;
      }
    });

    clearSourceMagazineMotion();
    placementMotionActive = false;
    placementSettleBarrier = false;
    placementStarted = shouldPlace;
    autoPlacementComplete = shouldPlace;
    section.classList.toggle('is-stage2-placement-started', shouldPlace);
    if (changed) rebuildLayerAssignments();
  }

  function requestPlacementContinuation() {
    var stageProgress;

    placementMotionActive = false;
    if (raf) return;
    raf = window.requestAnimationFrame(function () {
      raf = null;
      stageProgress = readSectionNumber('--stage2-p', stageProgressFromScrollProgress(latestScrollProgress));
      applyProgress(scrollProgressFromStageProgress(stageProgress));
    });
  }

  function pausePlacementForSettle() {
    placementSettleBarrier = true;
    requestPlacementContinuation();
  }

  function settleLayerInMoveTarget(layer, targetParent) {
    if (!layer || !targetParent) return;

    if (targetParent === tensorStack && !layer.getAttribute('data-stage2-assigned-node')) {
      insertLayerInTensorOrder(layer);
      return;
    }

    if (targetParent.hasAttribute('data-stage2-node-slots')) {
      insertLayerInNodeOrder(layer, targetParent);
      return;
    }

    targetParent.appendChild(layer);
  }

  function settleLayerPlacementState(layer, targetParent, nodeId) {
    if (!layer || !targetParent) return;

    if (nodeId) {
      layer.setAttribute('data-stage2-assigned-node', nodeId);
      setElementVar(layer, '--stage2-layer-compact-progress', '1');
      setElementVar(layer, '--stage2-layer-placed-opacity', '1');
      setElementVar(layer, '--stage2-layer-node-color', colorForNode(nodeId) || '#3b82f6');
      setLayerPlacedSemantics(layer, nodeId);
    } else {
      layer.removeAttribute('data-stage2-assigned-node');
      setElementVar(layer, '--stage2-layer-compact-progress', '0');
      clearLayerPlacementStyles(layer);
      setLayerSourceSemantics(layer);
    }

    settleLayerInMoveTarget(layer, targetParent);
  }

  function settleDormantFlightLayers(stageProgress) {
    var settled = false;

    if (!stage2FlightLayer || !stage2FlightLayer.isConnected) return false;

    modelLayers.forEach(function (layer) {
      var assignedNode;
      var targetParent;

      if (layer.parentElement !== stage2FlightLayer) return;
      if (stageProgress < PHASES[LOCK_PHASE_INDEX].start && stageProgress >= PHASES[MESH_PHASE_INDEX].start) return;

      assignedNode = layer.getAttribute('data-stage2-assigned-node');
      targetParent = assignedNode ? slotForNode(assignedNode) : tensorStack;
      if (!targetParent) return;

      if (window.anime && window.anime.remove && layer.__stage2MoveState) {
        window.anime.remove(layer.__stage2MoveState);
      }

      settleLayerInMoveTarget(layer, targetParent);
      clearLayerFlightStyles(layer);
      layer.__stage2MoveState = null;
      layer.__stage2MoveTargetParent = null;
      settled = true;
    });

    if (settled) placementMotionActive = false;

    return settled;
  }

  function removePlacementReservation(state) {
    if (!state || !state.reservation) return;
    if (state.reservation.parentElement) state.reservation.parentElement.removeChild(state.reservation);
    state.reservation = null;
  }

  function clearSourceMagazineMotion() {
    modelLayers.forEach(function (layer) {
      if (layer.parentElement !== tensorStack) return;
      layer.style.translate = '';
    });
  }

  function createPlacementReservation(layer, fromRect) {
    var reservation;
    var layerStyle;

    if (!layer || layer.parentElement !== tensorStack) return null;

    layerStyle = window.getComputedStyle(layer);
    reservation = document.createElement('div');
    reservation.className = 'stage2-layer-slot-reservation';
    reservation.setAttribute('data-stage2-placement-reservation', layerIdFor(layer));
    reservation.style.flex = '0 0 ' + Math.round(fromRect.height * 100) / 100 + 'px';
    reservation.style.height = Math.round(fromRect.height * 100) / 100 + 'px';
    reservation.style.minHeight = Math.round(fromRect.height * 100) / 100 + 'px';
    reservation.style.marginTop = layerStyle.marginTop;

    tensorStack.insertBefore(reservation, layer);
    return reservation;
  }

  function syncSourceLayerHeightFromRect(fromRect) {
    var height;

    if (!fromRect) return;

    height = Math.max(28, Math.round(fromRect.height * 100) / 100);
    setVar('--stage2-source-layer-height', height + 'px');
  }

  function sourceStepForLayer(index, fromRect) {
    var nextLayer = modelLayers[index + 1];
    var nextRect;

    if (nextLayer && nextLayer.parentElement === tensorStack) {
      nextRect = nextLayer.getBoundingClientRect();
      return Math.max(fromRect.height, nextRect.top - fromRect.top);
    }

    return fromRect.height;
  }

  function applySourceMagazineMotion(sourceIndex, distance, progress) {
    var offset = -Math.round(distance * easeInOut(progress) * 100) / 100;

    modelLayers.forEach(function (layer, index) {
      if (layer.parentElement !== tensorStack) return;
      layer.style.translate = index > sourceIndex ? '0 ' + offset + 'px' : '';
    });
  }

  function settleLayerSourceState(layer, state) {
    if (!layer || !tensorStack) return;

    layer.removeAttribute('data-stage2-assigned-node');
    setElementVar(layer, '--stage2-layer-compact-progress', '0');
    clearLayerPlacementStyles(layer);
    setLayerSourceSemantics(layer);

    if (state && state.reservation && state.reservation.parentElement === tensorStack) {
      tensorStack.insertBefore(layer, state.reservation);
      removePlacementReservation(state);
      return;
    }

    insertLayerInTensorOrder(layer);
  }

  function arePriorPlacementLayersSettled(index) {
    var i;
    var layer;
    var targetNode;
    var targetParent;

    for (i = 0; i < index; i += 1) {
      layer = modelLayers[i];
      targetNode = DEFAULT_ASSIGNMENTS[layerIdFor(layer)] || null;
      targetParent = targetNode ? slotForNode(targetNode) : null;

      if (!targetNode || !targetParent) continue;
      if (layer.__stage2PlacementData) return false;
      if (layer.parentElement === stage2FlightLayer) return false;
      if (layer.getAttribute('data-stage2-assigned-node') !== targetNode) return false;
      if (layer.parentElement !== targetParent) return false;
    }

    return true;
  }

  function placementProgressForLayer(index, phaseLocal) {
    var threshold = PLACEMENT_LAYER_START + index * PLACEMENT_LAYER_WINDOW;
    var progress = (phaseLocal - threshold) / PLACEMENT_LAYER_WINDOW;

    if (progress <= PLACEMENT_PROGRESS_EPSILON) return 0;
    if (progress >= 1 - PLACEMENT_PROGRESS_EPSILON) return 1;
    return clamp(progress, 0, 1);
  }

  function restoreSettledSourceLayersBeforeActivePlacement(phaseLocal, flightLayer) {
    var restored = false;

    modelLayers.forEach(function (layer, index) {
      var targetNode;
      var targetParent;
      var currentNode;

      if (layer.__stage2PlacementData) return;

      targetNode = DEFAULT_ASSIGNMENTS[layerIdFor(layer)] || null;
      targetParent = targetNode ? slotForNode(targetNode) : null;
      if (!targetNode || !targetParent) return;
      if (placementProgressForLayer(index, phaseLocal) > 0) return;

      currentNode = layer.getAttribute('data-stage2-assigned-node') || null;
      if (!currentNode && layer.parentElement !== flightLayer) return;

      settleLayerPlacementState(layer, tensorStack, null);
      restored = true;
    });

    if (!restored) return false;

    clearSourceMagazineMotion();
    placementMotionActive = false;
    placementSettleBarrier = false;
    rebuildLayerAssignments();
    return true;
  }

  function animateLayerFromRect(layer, fromRect, options, delay) {
    var duration = Math.max(0, Number(options && options.duration) || 0);
    var ease = options && options.ease ? options.ease : 'inOutExpo';
    var flightLayer;
    var flightRect;
    var targetParent;
    var toRect;
    var state;
    var fromLeft;
    var fromTop;
    var toLeft;
    var toTop;

    if (!layer || !fromRect) return;

    flightLayer = ensureStage2FlightLayer();
    if (!flightLayer) return;

    toRect = options && options.toRect ? options.toRect : layer.getBoundingClientRect();
    targetParent = layer.parentElement;
    flightRect = flightLayer.getBoundingClientRect();
    fromLeft = fromRect.left - flightRect.left;
    fromTop = fromRect.top - flightRect.top;
    toLeft = toRect.left - flightRect.left;
    toTop = toRect.top - flightRect.top;

    if (window.anime && window.anime.remove && layer.__stage2MoveState) {
      window.anime.remove(layer.__stage2MoveState);
    }

    if (Math.abs(fromLeft - toLeft) < 0.5 && Math.abs(fromTop - toTop) < 0.5) {
      clearLayerFlightStyles(layer);
      layer.__stage2MoveState = null;
      requestPlacementContinuation();
      return;
    }

    state = {
      left: fromLeft,
      top: fromTop,
      width: fromRect.width,
      height: fromRect.height,
      visualProgress: 0,
    };
    layer.__stage2MoveState = state;
    layer.__stage2MoveTargetParent = targetParent;
    flightLayer.appendChild(layer);
    setLayerFlightVisuals(layer, 0);
    layer.classList.add('is-stage2-layer-in-flight');
    layer.style.position = 'absolute';
    layer.style.left = Math.round(fromLeft * 100) / 100 + 'px';
    layer.style.top = Math.round(fromTop * 100) / 100 + 'px';
    layer.style.width = Math.round(fromRect.width * 100) / 100 + 'px';
    layer.style.height = Math.round(fromRect.height * 100) / 100 + 'px';
    layer.style.marginTop = '0px';
    layer.style.transform = 'translate3d(0, var(--stage2-flight-lift-y, 0px), 0) scale(var(--stage2-flight-scale, 1))';
    layer.style.translate = 'none';
    layer.style.willChange = 'left, top, width, height, transform, box-shadow';

    if (!window.anime || typeof window.anime.animate !== 'function' || !duration) {
      settleLayerInMoveTarget(layer, targetParent);
      clearLayerFlightStyles(layer);
      layer.__stage2MoveState = null;
      layer.__stage2MoveTargetParent = null;
      requestPlacementContinuation();
      return;
    }

    window.anime.animate(state, {
      left: toLeft,
      top: toTop,
      width: toRect.width,
      height: fromRect.height,
      visualProgress: 1,
      duration: duration,
      delay: delay || 0,
      ease: ease,
      onUpdate: function () {
        if (layer.__stage2MoveState !== state) return;
        layer.style.left = Math.round(state.left * 100) / 100 + 'px';
        layer.style.top = Math.round(state.top * 100) / 100 + 'px';
        layer.style.width = Math.round(state.width * 100) / 100 + 'px';
        layer.style.height = Math.round(state.height * 100) / 100 + 'px';
        setLayerFlightVisuals(layer, state.visualProgress);
      },
      onComplete: function () {
        if (layer.__stage2MoveState !== state) return;
        settleLayerInMoveTarget(layer, layer.__stage2MoveTargetParent);
        clearLayerFlightStyles(layer);
        layer.__stage2MoveState = null;
        layer.__stage2MoveTargetParent = null;
        requestPlacementContinuation();
        if (typeof options.onComplete === 'function') options.onComplete();
      },
    });
  }

  function ensureAutoPlacement(stageProgress) {
    var phaseLocal;
    var flightLayer;
    var hasActiveTransition;
    var i;
    var layer;
    var state;
    var currentNode;
    var targetNode;
    var progress;
    var eased;
    var fromRect;
    var flightRect;
    var targetParent;
    var fromLeft;
    var fromTop;
    var toRect;
    var toLeft;
    var toTop;
    var toWidth;
    var toHeight;
    var sourceStep;
    var reservation;
    var restoredSourceLayer;

    if (reduceMotion || !tensorStack || !modelLayers.length) return;

    phaseLocal = phaseProgress(PHASES[MESH_PHASE_INDEX], stageProgress);
    flightLayer = ensureStage2FlightLayer();
    hasActiveTransition = false;

    if (stageProgress >= PHASES[LOCK_PHASE_INDEX].start && settleAllPlacementTargets()) return;

    if (placementSettleBarrier) {
      placementSettleBarrier = false;
      placementMotionActive = false;
      rebuildLayerAssignments();
      if (!hasActiveLayerFlight()) return;
    }

    for (i = 0; i < modelLayers.length; i += 1) {
      layer = modelLayers[i];
      state = layer.__stage2PlacementData;
      if (!state) continue;
      hasActiveTransition = true;

      progress = placementProgressForLayer(i, phaseLocal);
      eased = easeInOut(progress);

      layer.style.left = Math.round(lerp(state.fromLeft, state.toLeft, eased) * 100) / 100 + 'px';
      layer.style.top = Math.round(lerp(state.fromTop, state.toTop, eased) * 100) / 100 + 'px';
      layer.style.width = Math.round(lerp(state.fromWidth, state.toWidth, eased) * 100) / 100 + 'px';
      layer.style.height = Math.round(state.fromHeight * 100) / 100 + 'px';
      setLayerFlightVisuals(layer, progress);

      if (typeof state.sourceIndex === 'number') {
        if (manualStageSeekActive) clearSourceMagazineMotion();
        else applySourceMagazineMotion(state.sourceIndex, state.sourceStep, progress);
      }

      if (progress >= 1) {
        settleLayerPlacementState(layer, state.targetParent, state.targetNode);
        removePlacementReservation(state);
        clearSourceMagazineMotion();
        clearLayerFlightStyles(layer);
        layer.__stage2PlacementData = null;
        rebuildLayerAssignments();
        pausePlacementForSettle();
        return;
      } else if (progress <= 0) {
        settleLayerSourceState(layer, state);
        clearSourceMagazineMotion();
        clearLayerFlightStyles(layer);
        layer.__stage2PlacementData = null;
        rebuildLayerAssignments();
        pausePlacementForSettle();
        return;
      } else {
        placementMotionActive = true;
        rebuildLayerAssignments();
        return;
      }
    }

    restoreSettledSourceLayersBeforeActivePlacement(phaseLocal, flightLayer);

    for (i = 0; i < modelLayers.length; i += 1) {
      layer = modelLayers[i];
      if (layer.__stage2PlacementData) continue;

      targetNode = DEFAULT_ASSIGNMENTS[layerIdFor(layer)] || null;
      targetParent = targetNode ? slotForNode(targetNode) : null;
      if (!targetNode || !targetParent) continue;

      progress = placementProgressForLayer(i, phaseLocal);
      currentNode = layer.getAttribute('data-stage2-assigned-node') || null;

      if (progress <= 0) {
        if (currentNode || layer.parentElement === flightLayer) {
          settleLayerPlacementState(layer, tensorStack, null);
          restoredSourceLayer = true;
        }
        continue;
      }

      if (!arePriorPlacementLayersSettled(i)) {
        if (restoredSourceLayer) break;
        return;
      }

      if (progress >= 1) {
        if (currentNode !== targetNode || layer.parentElement === flightLayer) {
          settleLayerPlacementState(layer, targetParent, targetNode);
          rebuildLayerAssignments();
          pausePlacementForSettle();
          return;
        }
        continue;
      }

      flightRect = flightLayer.getBoundingClientRect();
      settleLayerPlacementState(layer, tensorStack, null);
      fromRect = layer.getBoundingClientRect();
      syncSourceLayerHeightFromRect(fromRect);
      sourceStep = sourceStepForLayer(i, fromRect);
      reservation = createPlacementReservation(layer, fromRect);

      fromLeft = fromRect.left - flightRect.left;
      fromTop = fromRect.top - flightRect.top;

      settleLayerPlacementState(layer, targetParent, targetNode);

      toRect = layer.getBoundingClientRect();
      toLeft = toRect.left - flightRect.left;
      toTop = toRect.top - flightRect.top;
      toWidth = toRect.width;
      toHeight = fromRect.height;

      flightLayer.appendChild(layer);
      setLayerFlightVisuals(layer, progress);
      layer.style.position = 'absolute';
      layer.style.left = Math.round(fromLeft * 100) / 100 + 'px';
      layer.style.top = Math.round(fromTop * 100) / 100 + 'px';
      layer.style.width = Math.round(fromRect.width * 100) / 100 + 'px';
      layer.style.height = Math.round(fromRect.height * 100) / 100 + 'px';
      layer.style.marginTop = '0px';
      layer.style.transform = 'translate3d(0, var(--stage2-flight-lift-y, 0px), 0) scale(var(--stage2-flight-scale, 1))';
      layer.style.translate = 'none';
      layer.style.willChange = 'left, top, width, height, transform, box-shadow';
      layer.classList.add('is-stage2-layer-in-flight');

      layer.__stage2PlacementData = {
        fromLeft: fromLeft,
        fromTop: fromTop,
        fromWidth: fromRect.width,
        fromHeight: fromRect.height,
        toLeft: toLeft,
        toTop: toTop,
        toWidth: toWidth,
        toHeight: toHeight,
        targetParent: targetParent,
        targetNode: targetNode,
        sourceIndex: i,
        sourceStep: sourceStep,
        reservation: reservation,
      };

      eased = easeInOut(progress);
      layer.style.left = Math.round(lerp(fromLeft, toLeft, eased) * 100) / 100 + 'px';
      layer.style.top = Math.round(lerp(fromTop, toTop, eased) * 100) / 100 + 'px';
      layer.style.width = Math.round(lerp(fromRect.width, toWidth, eased) * 100) / 100 + 'px';
      layer.style.height = Math.round(fromRect.height * 100) / 100 + 'px';
      setLayerFlightVisuals(layer, progress);
      if (manualStageSeekActive) clearSourceMagazineMotion();
      else applySourceMagazineMotion(i, sourceStep, progress);

      hasActiveTransition = true;
      cancelLayoutClearTimers();
      placementMotionActive = true;
      rebuildLayerAssignments();
      return;
    }

    if (restoredSourceLayer) {
      clearSourceMagazineMotion();
      placementMotionActive = false;
      placementSettleBarrier = false;
      rebuildLayerAssignments();
      return;
    }

    if (!hasActiveTransition) {
      if (settleDormantFlightLayers(stageProgress)) {
        rebuildLayerAssignments();
        return;
      }
      placementMotionActive = false;
      clearSourceMagazineMotion();
    } else {
      placementMotionActive = true;
    }

    rebuildLayerAssignments();
  }

  function applyReducedStaticPlacement() {
    if (!tensorStack || !modelLayers.length) return;

    modelLayers.forEach(function (layer) {
      var layerId = layer.getAttribute('data-stage2-model-layer');
      var nodeId = DEFAULT_ASSIGNMENTS[layerId];
      var slots = slotForNode(nodeId);

      if (!slots) return;
      layer.style.transform = '';
      layer.style.translate = '';
      slots.appendChild(layer);
      layerAssignments[layerId] = nodeId;
      layer.setAttribute('data-stage2-assigned-node', nodeId);
      setLayerReducedSemantics(layer, nodeId);
    });

    placementStarted = true;
    autoPlacementComplete = true;
    section.classList.add('is-stage2-placement-started');
  }

  function initTimeline() {
    if (!window.anime || !window.anime.createTimeline) return;

    timeline = window.anime.createTimeline({
      autoplay: false,
      defaults: { ease: 'linear' },
    });

    timeline.add(timelineState, {
      progress: [0, 1],
      duration: 1000,
      ease: 'linear',
      onUpdate: function () {
        if (isScrubbing || hasManualStageSeekOverride()) return;
        applyProgress(timelineState.progress);
      },
    }, 0);

    if (window.anime.onScroll) {
      try {
        scrollObserver = window.anime.onScroll({
          target: section,
          container: document.body,
          enter: 'top top',
          leave: 'bottom bottom',
          sync: scrollSmoothSync(),
        });
        if (scrollObserver && typeof scrollObserver.link === 'function') {
          scrollObserver.link(timeline);
          usesScrollObserver = true;
        }
      } catch (_error) {
        scrollObserver = null;
        usesScrollObserver = false;
      }
    }
  }

  function buildScrubberTicks(track) {
    var ticks = track && track.querySelector('.stage2-scrubber-ticks');
    var majors = track && track.querySelector('.stage2-scrubber-majors');
    var i;
    var tick;
    if (!ticks || !majors || ticks.children.length || majors.children.length) return;

    for (i = 0; i < scrubberTickCount; i += 1) {
      tick = document.createElement('span');
      tick.className = 'stage2-scrubber-tick';
      tick.style.setProperty('--tick-left', tickPosition(i / (scrubberTickCount - 1)));
      ticks.appendChild(tick);
    }

    PHASES.forEach(function (phase) {
      var major = document.createElement('span');
      major.className = 'stage2-scrubber-major';
      major.style.setProperty('--tick-left', tickPosition(scrubberProgressFromProgress(phase.start)));
      major.setAttribute('title', phase.label);
      majors.appendChild(major);
    });

    var endMajor = document.createElement('span');
    endMajor.className = 'stage2-scrubber-major';
    endMajor.style.setProperty('--tick-left', tickPosition(1));
    endMajor.setAttribute('title', 'aperture close');
    majors.appendChild(endMajor);
  }

  function initScrubber() {
    var scrubber = section.querySelector('.stage2-scrubber');
    var track = section.querySelector('.stage2-scrubber-track');
    var pendingScrubberProgress = null;
    var scrubberTrackRect = null;
    scrubInput = section.querySelector('.stage2-scrubber-input');

    if (track) buildScrubberTicks(track);

    function measureScrubberTrack() {
      scrubberTrackRect = track ? track.getBoundingClientRect() : null;
      return scrubberTrackRect;
    }

    function invalidateScrubberTrack() {
      scrubberTrackRect = null;
    }

    function progressFromClientX(clientX) {
      var rect = scrubberTrackRect || measureScrubberTrack();
      return rect && rect.width ? clamp((clientX - rect.left) / rect.width, 0, 1) : 0;
    }

    function applyScrubberProgress(scrubberProgress) {
      scrubberProgress = clamp(scrubberProgress, 0, 1);
      if (scrubInput) scrubInput.value = Math.round(scrubberProgress * 1000);
      seekStageProgressImmediate(progressFromScrubberProgress(scrubberProgress));
    }

    function cancelScheduledScrubberSeek() {
      if (scrubberSeekRaf) window.cancelAnimationFrame(scrubberSeekRaf);
      scrubberSeekRaf = null;
      pendingScrubberProgress = null;
    }

    function flushScheduledScrubberSeek() {
      if (pendingScrubberProgress === null) return;
      if (scrubberSeekRaf) window.cancelAnimationFrame(scrubberSeekRaf);
      scrubberSeekRaf = null;
      applyScrubberProgress(pendingScrubberProgress);
      pendingScrubberProgress = null;
    }

    function scheduleScrubberProgress(scrubberProgress, immediate) {
      pendingScrubberProgress = clamp(scrubberProgress, 0, 1);

      if (immediate) {
        flushScheduledScrubberSeek();
        return;
      }

      if (scrubberSeekRaf) return;
      scrubberSeekRaf = window.requestAnimationFrame(function () {
        scrubberSeekRaf = null;
        if (pendingScrubberProgress === null) return;
        applyScrubberProgress(pendingScrubberProgress);
        pendingScrubberProgress = null;
      });
    }

    function setScrubberFromClientX(clientX, immediate) {
      if (!track) return;
      scheduleScrubberProgress(progressFromClientX(clientX), immediate);
    }

    function endScrubbing() {
      flushScheduledScrubberSeek();
      isScrubbing = false;
      scrubberPointerId = null;
      scrubberTrackRect = null;
      if (scrubber) scrubber.classList.remove('is-scrubbing');
    }

    if (scrubInput) {
      scrubInput.addEventListener('input', function () {
        scheduleScrubberProgress(Number(scrubInput.value) / 1000, false);
      });
    }

    if (scrubber && window.PointerEvent) {
      scrubber.addEventListener('pointerdown', function (event) {
        if (event.button !== undefined && event.button !== 0) return;

        isScrubbing = true;
        scrubberPointerId = event.pointerId;
        measureScrubberTrack();
        scrubber.classList.add('is-scrubbing');
        if (scrubber.setPointerCapture) scrubber.setPointerCapture(event.pointerId);
        setScrubberFromClientX(event.clientX, true);
        event.preventDefault();
      });

      scrubber.addEventListener('pointermove', function (event) {
        if (!isScrubbing || event.pointerId !== scrubberPointerId) return;

        setScrubberFromClientX(event.clientX, false);
        event.preventDefault();
      });

      scrubber.addEventListener('pointerup', function (event) {
        if (event.pointerId !== scrubberPointerId) return;
        if (scrubber.releasePointerCapture) scrubber.releasePointerCapture(event.pointerId);
        endScrubbing();
      });

      scrubber.addEventListener('pointercancel', function (event) {
        if (event.pointerId !== scrubberPointerId) return;
        endScrubbing();
      });

      window.addEventListener('pointerup', function () {
        if (isScrubbing) endScrubbing();
      }, { passive: true });

      window.addEventListener('scroll', invalidateScrubberTrack, { passive: true });
      window.addEventListener('resize', invalidateScrubberTrack, { passive: true });
      window.addEventListener('orientationchange', invalidateScrubberTrack, { passive: true });

      window.addEventListener('pagehide', function () {
        cancelScheduledScrubberSeek();
        if (isScrubbing) endScrubbing();
      }, { passive: true });

      document.addEventListener('visibilitychange', function () {
        if (!document.hidden) return;
        cancelScheduledScrubberSeek();
        if (isScrubbing) endScrubbing();
      });
    }
  }

  function initLayerInteractions() {
    modelLayers = Array.prototype.slice.call(section.querySelectorAll('[data-stage2-model-layer]'));
    if (!modelLayers.length) return;

    modelLayers.forEach(function (layer) {
      setLayerSourceSemantics(layer);
    });
  }

  function targetForAnchor(link) {
    var id;

    if (!link || !link.hash || link.hash === '#') return null;

    try {
      id = decodeURIComponent(link.hash.slice(1));
    } catch (_error) {
      id = link.hash.slice(1);
    }

    return id ? document.getElementById(id) : null;
  }

  function pushAnchorHash(hash) {
    if (!hash) return;

    if (window.history && typeof window.history.pushState === 'function') {
      window.history.pushState(null, '', hash);
      return;
    }

    window.location.hash = hash.slice(1);
  }

  function restoreScrollBehaviorSoon() {
    window.requestAnimationFrame(function () {
      restoreScrollBehavior();
    });
  }

  function scrollToAnchorTarget(target) {
    var scrollMarginTop;

    if (!target) return;

    scrollMarginTop = Number.parseFloat(window.getComputedStyle(target).scrollMarginTop) || 0;
    window.scrollTo({ top: window.scrollY + target.getBoundingClientRect().top - scrollMarginTop, behavior: 'auto' });
  }

  function lockCompletedStageAtAnchor(target) {
    scrollToAnchorTarget(target);
    beginManualStageSeekOverride();
    applyProgress(1);
  }

  function initLearnMore() {
    var link = document.querySelector('.hero-learn');
    if (!link) return;

    link.addEventListener('click', function (event) {
      var target;

      if (event.metaKey || event.ctrlKey || event.shiftKey || event.altKey || event.button) return;

      target = targetForAnchor(link);
      if (!target) return;

      event.preventDefault();
      cancelScrubScrollAnimation();
      cancelScrollSmoothing();
      clearManualStageSeekOverride();
      if (window.anime && window.anime.remove) window.anime.remove(smoothState);
      document.body.classList.add('is-hero-complete');
      document.body.classList.add('is-stage2-complete');
      document.body.classList.remove('is-stage2-active');
      document.body.classList.remove('home-intro');
      forceInstantScrollBehavior();
      lockCompletedStageAtAnchor(target);
      pushAnchorHash(link.hash);
      window.requestAnimationFrame(function () {
        lockCompletedStageAtAnchor(target);
        window.requestAnimationFrame(function () {
          lockCompletedStageAtAnchor(target);
          restoreScrollBehaviorSoon();
        });
      });
    });
  }

  function exposeStage2DebugControls() {
    window.__meshStage2 = {
      seek: function (progress) {
        clearManualStageSeekOverride();
        cancelScrubScrollAnimation();
        scrollToProgress(progress);
        smoothTo(progress, true);
      },
      seekStage: function (progress) {
        seekStageProgressImmediate(clamp(progress, 0, SCRUBBER_END));
      },
      scrubToPhaseTwo: function () {
        scrubToStageProgress(LEARN_MORE_TARGET_PROGRESS);
      },
      debug: function () {
        return {
          hasLayout: Boolean(stage2Layout),
          assignments: Object.assign({}, layerAssignments),
          placedCount: modelLayers.filter(function (layer) {
            return Boolean(layer.getAttribute('data-stage2-assigned-node'));
          }).length,
        };
      },
    };
  }

  function init() {
    section = document.querySelector('[data-stage2-scroll]');
    if (!section) return;

    sticky = section.querySelector('.hero-stage-sticky');
    nav = document.querySelector('nav.top');
    heroViz = section.querySelector('.hero-viz');
    heroTitle = section.querySelector('.hero-title-loop');
    heroFooter = section.querySelector('.hero-footer');
    finalHookLines = Array.prototype.slice.call(section.querySelectorAll('[data-stage2-final-line]'));
    serverPlate = section.querySelector('.hero-viz [data-node-id="server"] .node-plate');
    tensorStack = section.querySelector('[data-stage2-tensor-stack]');
    layoutRoot = section.querySelector('[data-stage2-layout-root]');
    meshNodes = Array.prototype.slice.call(section.querySelectorAll('[data-stage2-node]'));
    phaseName = section.querySelector('.stage2-phase-name');
    phaseIndex = section.querySelector('.stage2-phase-index');
    phasePanels = Array.prototype.slice.call(section.querySelectorAll('.stage2-phase-panel'));

    if (!reduceMotion && typeof window.anime === 'undefined') {
      window.requestAnimationFrame(init);
      return;
    }

    initTimeline();
    syncHeroChromeMetrics();
    initStage2Layout();
    initSliceTimeline();
    initFinalHookWords();
    renderStage2NodeIcons();
    initScrubber();
    initLayerInteractions();
    initLearnMore();
    exposeStage2DebugControls();

    if (reduceMotion) {
      applyProgress(scrollProgressFromStageProgress(0.805));
      setVar('--stage2-model-shrink-progress', 1);
      setVar('--stage2-placement-progress', 1);
      setVar('--stage2-lock-progress', 1);
      setVar('--stage2-lock-card-progress', 1);
      setVar('--stage2-lock-mark-progress', 1);
      setVar('--stage2-lock-check-progress', 1);
      setVar('--stage2-lock-text-progress', 1);
      setVar('--stage2-lock-route-progress', 1);
      setVar('--stage2-lock-node-progress', 1);
      applyReducedStaticPlacement();
      syncScrubberAccessibility(0);
      document.body.classList.remove('is-stage2-active');
      section.classList.add('is-stage2-reduced-summary');
      section.classList.remove('is-stage2-title-cleared', 'is-stage2-footer-cleared');
      if (nav) {
        nav.style.opacity = '';
        nav.style.transform = '';
        nav.style.pointerEvents = '';
      }
      if (heroTitle) {
        heroTitle.style.opacity = '';
        heroTitle.style.transform = '';
      }
      if (heroFooter) {
        heroFooter.style.opacity = '';
        heroFooter.style.transform = '';
        heroFooter.style.pointerEvents = '';
      }
      return;
    }

    if (!usesScrollObserver) {
      window.addEventListener('scroll', syncTitleFromScroll, { passive: true });
      window.addEventListener('scroll', function () { requestUpdate(false); }, { passive: true });
    }
    window.addEventListener('resize', handleGeometryChange);
    window.addEventListener('orientationchange', handleGeometryChange, { passive: true });
    window.addEventListener('mesh:layout', handleGeometryChange);
    window.addEventListener('wheel', clearManualStageSeekOverrideOnUserIntent, { passive: true });
    window.addEventListener('touchstart', clearManualStageSeekOverrideOnUserIntent, { passive: true });
    window.addEventListener('keydown', clearManualStageSeekOverrideOnUserIntent);
    window.addEventListener('scroll', requestCompletedGridClip, { passive: true });
    if (usesScrollObserver) {
      applyProgress(progressFromScroll());
    } else {
      requestUpdate(true);
    }
    if (usesScrollObserver) {
      window.requestAnimationFrame(function () {
        syncServerGeometry();
        syncMeshGeometry();
      });
    }

    var params = new URLSearchParams(window.location.search);
    var forcedProgress = params.has('stage2Progress') ? Number(params.get('stage2Progress')) : NaN;
    if (!Number.isNaN(forcedProgress)) {
      window.setTimeout(function () {
        scrollToProgress(forcedProgress);
        smoothTo(forcedProgress, true);
      }, 240);
    }

  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
