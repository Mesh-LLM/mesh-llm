import markdownItAnchor from "markdown-it-anchor";

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

  eleventyConfig.addPassthroughCopy("src/mesh-llm-logo.svg");
  eleventyConfig.addPassthroughCopy("src/CNAME");
  eleventyConfig.addPassthroughCopy("src/assets");
  eleventyConfig.addPassthroughCopy({ "../install.sh": "install.sh" });
  eleventyConfig.addPassthroughCopy({ "../install.ps1": "install.ps1" });

  eleventyConfig.amendLibrary("md", (md) => {
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
