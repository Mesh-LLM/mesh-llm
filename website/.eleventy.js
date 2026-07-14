import hljs from "highlight.js/lib/core";
import ini from "highlight.js/lib/languages/ini";
import bash from "highlight.js/lib/languages/bash";
import json from "highlight.js/lib/languages/json";
import rust from "highlight.js/lib/languages/rust";
import markdownItAnchor from "markdown-it-anchor";

hljs.registerLanguage("ini", ini);
hljs.registerLanguage("bash", bash);
hljs.registerLanguage("json", json);
hljs.registerLanguage("rust", rust);

const decodeHtmlEntities = (value) =>
  value
    .replace(/&amp;/g, "&")
    .replace(/&lt;/g, "<")
    .replace(/&gt;/g, ">")
    .replace(/&quot;/g, '"')
    .replace(/&#39;/g, "'");

const headingText = (value) =>
  decodeHtmlEntities(
    value
      .replace(/<[^>]*>/g, "")
      .replace(/\s+/g, " ")
      .trim()
  );

export default function(eleventyConfig) {
  eleventyConfig.setServerOptions({
    showAllHosts: true,
  });

  eleventyConfig.addPassthroughCopy("src/funding.json");
  eleventyConfig.addPassthroughCopy("src/.well-known");
  eleventyConfig.addPassthroughCopy("src/mesh-llm-logo.svg");
  eleventyConfig.addPassthroughCopy("src/CNAME");
  eleventyConfig.addPassthroughCopy("src/assets");
  eleventyConfig.addPassthroughCopy({ "../install.sh": "install.sh" });
  eleventyConfig.addPassthroughCopy({ "../install.ps1": "install.ps1" });

  eleventyConfig.amendLibrary("md", (md) => {
    md.set({
      highlight: (str, lang) => {
        const langMap = { toml: "ini" };
        const hl = lang && langMap[lang] ? langMap[lang] : lang;
        if (hl && hljs.getLanguage(hl)) {
          try {
            return `<pre class="language-${hl}"><code class="language-${hl}">${hljs.highlight(str, { language: hl, ignoreIllegals: true }).value}</code></pre>`;
          } catch (e) {
            console.debug("highlight.js error for lang=%s: %s", hl, e);
          }
        }
        if (lang === "mermaid") {
          return `<pre class="language-mermaid"><code class="language-mermaid">${md.utils.escapeHtml(str)}</code></pre>`;
        }
        return `<pre><code>${md.utils.escapeHtml(str)}</code></pre>`;
      },
    });
    md.use(markdownItAnchor, {
      permalink: false,
      slugify: (value) =>
        String(value)
          .trim()
          .toLowerCase()
          .replace(/[^a-z0-9]+/g, "-")
          .replace(/(^-|-$)/g, ""),
    });
  });

  eleventyConfig.addFilter("json", (value) => JSON.stringify(value));
  eleventyConfig.addFilter("tocHeadings", (content) => {
    if (typeof content !== "string") return [];

    return Array.from(content.matchAll(/<h2\s+[^>]*id="([^"]+)"[^>]*>([\s\S]*?)<\/h2>/g)).map(
      ([, id, text]) => ({ id, text: headingText(text) })
    );
  });
  eleventyConfig.addFilter("urlPath", (url) => {
    if (!url) return url;
    const hashIndex = url.indexOf("#");
    return hashIndex !== -1 ? url.substring(0, hashIndex) : url;
  });
  eleventyConfig.addFilter("format", (fmt, ...args) => {
    let i = 0;
    return fmt.replace(/%(\d+)?([dx])/g, (_, width, type) => {
      const val = String(args[i++] ?? 0);
      if (type === "d" && width) return val.padStart(Number(width), "0");
      return val;
    });
  });
  eleventyConfig.addTransform("trim-trailing-whitespace", (content) =>
    typeof content === "string" ? content.replace(/[ \t]+$/gm, "") : content
  );

  return {
    dir: {
      input: "src",
      includes: "_includes",
      output: "../docs",
    },
    markdownTemplateEngine: "njk",
    htmlTemplateEngine: "njk",
    templateFormats: ["md", "njk", "html"],
  };
}
