async function renderMermaid() {
  const blocks = document.querySelectorAll("pre > code.language-mermaid");

  if (!blocks.length) {
    return;
  }

  try {
    const { default: mermaid } = await import(
      "https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.esm.min.mjs"
    );

    mermaid.initialize({
      startOnLoad: false,
      securityLevel: "loose",
      theme: "dark",
      flowchart: {
        curve: "basis",
        htmlLabels: true,
      },
      sequence: {
        useMaxWidth: true,
      },
    });

    const nodes = [];

    blocks.forEach((code) => {
      const container = document.createElement("div");
      const pre = code.parentElement;
      const frame = pre?.parentElement?.classList.contains("code-copy-frame")
        ? pre.parentElement
        : pre;

      container.className = "mermaid";
      container.setAttribute("role", "img");
      container.setAttribute("aria-label", "Diagram");
      container.textContent = code.textContent;
      frame?.replaceWith(container);
      nodes.push(container);
    });

    await mermaid.run({ nodes });
  } catch (err) {
    console.error("Mermaid rendering failed:", err);
  }
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", renderMermaid, { once: true });
} else {
  renderMermaid();
}
