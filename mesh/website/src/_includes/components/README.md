# Reusable Website Components

Eleventy templates in this directory hold the reusable structural pieces for the Mesh website.

- `ui.njk`: Small Nunjucks macros for section heads, badges, meters, topology cards, and architecture nodes.
- `stats-grid.njk`: Shared stat grid backed by `src/_data/homeStats.js`.
- `plugin-architecture.njk`: Plugin architecture diagram shared by homepage surfaces.
- `document-head.njk`: Common document metadata, favicon, and stylesheet links used by page layouts.

Use these for repeated structures with the same intent. Keep section-specific diagrams and copy in their owning section unless a pattern appears across multiple surfaces.
