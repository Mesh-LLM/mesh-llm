import { useEffect, useRef, useState } from "react";
import { Loader2 } from "lucide-react";

// Mermaid diagram renderer — loads mermaid from CDN on first use
const mermaidPromise = import(
  "https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.esm.min.mjs" as string
)
  .then((m) => {
    m.default.initialize({
      startOnLoad: false,
      theme: "dark",
      securityLevel: "antiscript",
    });
    return m.default;
  })
  .catch(() => null);

export function MermaidBlock({ code }: { code: string }) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [svg, setSvg] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!svg || !containerRef.current) return;
    containerRef.current.innerHTML = svg;
    return () => {
      if (containerRef.current) containerRef.current.innerHTML = "";
    };
  }, [svg]);

  useEffect(() => {
    let cancelled = false;
    setError(null);
    setSvg(null);
    mermaidPromise.then(async (mermaid) => {
      if (cancelled || !mermaid) {
        if (!cancelled) setError("Mermaid failed to load");
        return;
      }
      try {
        const id = `mermaid-${Date.now()}-${Math.random().toString(36).slice(2)}`;
        const { svg: rendered } = await mermaid.render(id, code);
        if (!cancelled) setSvg(rendered);
      } catch (e: unknown) {
        if (!cancelled)
          setError(e instanceof Error ? e.message : "Render failed");
      }
    });
    return () => {
      cancelled = true;
    };
  }, [code]);

  if (error)
    return (
      <pre className="my-2 rounded-lg border border-border/70 bg-background/80 p-3 text-xs text-muted-foreground">
        <code>{code}</code>
      </pre>
    );
  if (!svg)
    return (
      <div className="my-2 flex items-center gap-2 rounded-lg border border-border/70 bg-background/80 p-3 text-xs text-muted-foreground">
        <Loader2 className="h-3 w-3 animate-spin" />
        Rendering diagram…
      </div>
    );

  return (
    <div
      ref={containerRef}
      className="my-2 overflow-x-auto rounded-lg border border-border/70 bg-background/80 p-3 [&_svg]:max-w-full"
    />
  );
}
