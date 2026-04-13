import { useEffect, useRef, useState } from "react";

// KaTeX math renderer — loads from CDN on first use
let katexCssLoaded = false;
const katexPromise = import(
  "https://cdn.jsdelivr.net/npm/katex@0.16/dist/katex.mjs" as string
)
  .then((m) => {
    if (!katexCssLoaded) {
      const link = document.createElement("link");
      link.rel = "stylesheet";
      link.href = "https://cdn.jsdelivr.net/npm/katex@0.16/dist/katex.min.css";
      document.head.appendChild(link);
      katexCssLoaded = true;
    }
    return m.default;
  })
  .catch(() => null);

export function KaTeXBlock({ math, display }: { math: string; display: boolean }) {
  const [rendered, setRendered] = useState(false);
  const blockRef = useRef<HTMLDivElement | null>(null);
  const inlineRef = useRef<HTMLSpanElement | null>(null);

  useEffect(() => {
    let cancelled = false;
    setRendered(false);
    katexPromise.then((katex) => {
      const container = display ? blockRef.current : inlineRef.current;
      if (cancelled || !katex || !container) return;
      container.innerHTML = "";
      try {
        katex.render(math, container, {
          displayMode: display,
          throwOnError: false,
        });
        if (!cancelled) setRendered(true);
      } catch {}
    });
    return () => {
      cancelled = true;
    };
  }, [math, display]);

  return display ? (
    <>
      <div ref={blockRef} className={rendered ? "my-2 overflow-x-auto" : "hidden"} />
      {!rendered && (
        <div className="my-2 overflow-x-auto text-sm">
          <code>{math}</code>
        </div>
      )}
    </>
  ) : (
    <>
      <span ref={inlineRef} className={rendered ? undefined : "hidden"} />
      {!rendered && <code>{math}</code>}
    </>
  );
}
