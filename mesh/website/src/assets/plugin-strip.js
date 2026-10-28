(function () {
  var STRIP_SELECTOR = "[data-plugin-strip]";
  var REDUCE_QUERY = "(prefers-reduced-motion: reduce)";
  var PX_PER_SECOND = 28;

  function getAnime() {
    return window.anime && window.anime.animate ? window.anime : null;
  }

  function setup(strip) {
    var animeApi = getAnime();
    var track = strip.querySelector("[data-plugin-strip-track]");
    var reduceMotion = window.matchMedia(REDUCE_QUERY);
    if (!animeApi || !track || reduceMotion.matches) return;

    var originals = Array.prototype.slice.call(track.children);
    originals.forEach(function (item) {
      var copy = item.cloneNode(true);
      copy.setAttribute("aria-hidden", "true");
      copy.querySelectorAll("a").forEach(function (link) { link.tabIndex = -1; });
      track.appendChild(copy);
    });

    // One loop = width of the original set, including the gap before the copies.
    var gap = parseFloat(window.getComputedStyle(track).columnGap) || 0;
    var distance = track.scrollWidth / 2 + gap / 2;
    strip.classList.add("is-rotating");

    var animation = animeApi.animate(track, {
      translateX: [0, -distance],
      duration: (distance / PX_PER_SECOND) * 1000,
      ease: "linear",
      loop: true,
    });

    var hovered = false;
    var focused = false;
    var visible = true;
    function sync() {
      if (hovered || focused || !visible) animation.pause();
      else animation.play();
    }

    strip.addEventListener("mouseenter", function () { hovered = true; sync(); });
    strip.addEventListener("mouseleave", function () { hovered = false; sync(); });
    strip.addEventListener("focusin", function () { focused = true; sync(); });
    strip.addEventListener("focusout", function () { focused = false; sync(); });

    if ("IntersectionObserver" in window) {
      new IntersectionObserver(function (entries) {
        visible = entries[0].isIntersecting;
        sync();
      }).observe(strip);
    }

    reduceMotion.addEventListener("change", function (event) {
      if (event.matches) {
        animation.pause();
        animation.seek(0);
        strip.classList.remove("is-rotating");
      } else {
        sync();
      }
    });
  }

  function init() {
    Array.prototype.forEach.call(document.querySelectorAll(STRIP_SELECTOR), setup);
  }

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init);
  else init();
})();
