(function () {
  'use strict';

  var TITLE_SELECTOR = [
    '#topology .sec-head',
    '#catalog .sec-head',
    '#plugins .sec-head',
  ].join(',');
  var REDUCE_QUERY = '(prefers-reduced-motion: reduce)';
  var REVEAL_LINGER_MS = 800;
  var REVEAL_DURATION = 1750;
  var REVEAL_EASE = 'cubic-bezier(0.22, 1, 0.36, 1)';
  var REVEAL_TRANSITION = [
    'opacity ' + REVEAL_DURATION + 'ms ' + REVEAL_EASE,
    'transform ' + REVEAL_DURATION + 'ms ' + REVEAL_EASE,
    'filter ' + REVEAL_DURATION + 'ms ' + REVEAL_EASE,
  ].join(', ');
  var reduceMotion = window.matchMedia(REDUCE_QUERY);
  var observer = null;
  var hiddenObserver = null;
  var visibilityFrame = null;
  var revealTimers = new WeakMap();
  var titles = [];

  function toArray(list) {
    return Array.prototype.slice.call(list || []);
  }

  function getAnime() {
    return window.anime && window.anime.animate ? window.anime : null;
  }

  function isUsableTitle(title) {
    return title && title.textContent && title.textContent.trim().length > 0;
  }

  function isHidden(title) {
    return !!title.closest('[hidden]');
  }

  function isInViewport(title) {
    var rect = title.getBoundingClientRect();
    var height = window.innerHeight || document.documentElement.clientHeight;
    var width = window.innerWidth || document.documentElement.clientWidth;

    return rect.bottom >= 0 && rect.right >= 0 && rect.top <= height && rect.left <= width;
  }

  function clearTitle(title) {
    cancelQueuedReveal(title);
    title.classList.remove('is-title-revealing');
    title.classList.add('is-title-revealed');
    title.style.opacity = '';
    title.style.transform = '';
    title.style.filter = '';
    title.style.transition = '';
  }

  function cancelQueuedReveal(title) {
    var timer = revealTimers.get(title);

    if (timer) {
      window.clearTimeout(timer);
      revealTimers.delete(title);
    }
  }

  function queueReveal(title) {
    if (!title || title.classList.contains('is-title-revealed') || title.classList.contains('is-title-revealing')) return;
    if (revealTimers.has(title)) return;

    revealTimers.set(title, window.setTimeout(function () {
      revealTimers.delete(title);

      if (isHidden(title) || !isInViewport(title)) return;
      if (observer) observer.unobserve(title);

      revealTitle(title);
    }, REVEAL_LINGER_MS));
  }

  function prepareTitle(title) {
    title.classList.add('title-reveal');
    title.style.opacity = '0';
    title.style.transform = 'translate3d(0, 0.65rem, 0)';
    title.style.filter = 'blur(3px)';
  }

  function revealTitle(title) {
    var animeApi = getAnime();
    var finalizer = null;
    var completed = false;

    if (!title || title.classList.contains('is-title-revealed') || title.classList.contains('is-title-revealing')) return;

    if (reduceMotion.matches || !animeApi) {
      clearTitle(title);
      return;
    }

    title.classList.add('is-title-revealing');

    function completeReveal() {
      if (completed) return;

      completed = true;
      window.clearTimeout(finalizer);
      clearTitle(title);
    }

    animeApi.animate({ progress: 0 }, {
      progress: 1,
      duration: REVEAL_DURATION,
      ease: animeApi.eases && animeApi.eases.outCubic ? animeApi.eases.outCubic : 'linear',
      composition: 'replace',
    });

    title.style.transition = REVEAL_TRANSITION;
    window.requestAnimationFrame(function () {
      window.requestAnimationFrame(function () {
        title.style.opacity = '1';
        title.style.transform = 'translate3d(0, 0, 0)';
        title.style.filter = 'blur(0px)';
      });
    });
    finalizer = window.setTimeout(completeReveal, REVEAL_DURATION + 80);
  }

  function revealVisibleHiddenTitles() {
    titles.forEach(function (title) {
      if (title.classList.contains('is-title-revealed')) {
        cancelQueuedReveal(title);
        return;
      }

      if (!isHidden(title) && isInViewport(title)) {
        queueReveal(title);
      } else {
        cancelQueuedReveal(title);
      }
    });
  }

  function scheduleVisibleTitleReveal() {
    if (visibilityFrame !== null) return;

    visibilityFrame = window.requestAnimationFrame(function () {
      visibilityFrame = null;
      revealVisibleHiddenTitles();
    });
  }

  function addVisibilityFallbacks() {
    window.addEventListener('scroll', scheduleVisibleTitleReveal, { passive: true });
    window.addEventListener('resize', scheduleVisibleTitleReveal);
    window.addEventListener('pageshow', scheduleVisibleTitleReveal);
  }

  function removeVisibilityFallbacks() {
    if (visibilityFrame !== null) {
      window.cancelAnimationFrame(visibilityFrame);
      visibilityFrame = null;
    }

    titles.forEach(cancelQueuedReveal);

    window.removeEventListener('scroll', scheduleVisibleTitleReveal);
    window.removeEventListener('resize', scheduleVisibleTitleReveal);
    window.removeEventListener('pageshow', scheduleVisibleTitleReveal);
  }

  function armObserver() {
    observer = new IntersectionObserver(function (entries) {
      entries.forEach(function (entry) {
        if (!entry.isIntersecting || isHidden(entry.target)) {
          cancelQueuedReveal(entry.target);
          return;
        }

        queueReveal(entry.target);
      });
    }, {
      rootMargin: '0px 0px -12% 0px',
      threshold: 0.16,
    });

    titles.forEach(function (title) {
      observer.observe(title);
    });

    hiddenObserver = new MutationObserver(revealVisibleHiddenTitles);
    hiddenObserver.observe(document.body, {
      attributes: true,
      attributeFilter: ['hidden', 'class', 'style'],
      subtree: true,
    });

    addVisibilityFallbacks();
  }

  function revealAll() {
    titles.forEach(clearTitle);
    if (observer) observer.disconnect();
    if (hiddenObserver) hiddenObserver.disconnect();
    removeVisibilityFallbacks();
  }

  function initTitleReveal() {
    titles = toArray(document.querySelectorAll(TITLE_SELECTOR)).filter(isUsableTitle);

    if (!titles.length) return;

    titles.forEach(prepareTitle);

    if (reduceMotion.matches || !getAnime() || !('IntersectionObserver' in window)) {
      revealAll();
      return;
    }

    armObserver();
    revealVisibleHiddenTitles();
    window.setTimeout(scheduleVisibleTitleReveal, 250);

    if (reduceMotion.addEventListener) {
      reduceMotion.addEventListener('change', function (event) {
        if (event.matches) revealAll();
      });
    }
  }

  document.addEventListener('DOMContentLoaded', initTitleReveal);
})();
