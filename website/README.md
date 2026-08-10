# Mesh LLM Website

The public website is built with Eleventy and emitted into `../docs` for static hosting.
The root-level `just website-build` recipe also generates the published Rust
crate API reference under `../docs/crates/`.

```sh
cd website
npm install
npm run build
```

During development:

```sh
cd website
npm run dev
```

From the repository root:

```sh
just website-dev
just website-build
```

Use `just crate-docs` to regenerate only the Rust crate API reference.

`npm run dev` runs Eleventy with watch mode, incremental builds, and browser
reload on port 8765.

Source files live in `website/src`:

- `index.njk` - landing page
- `catalog/index.njk` - live Hugging Face catalog page
- `docs/index.njk` - docs landing page
- `crates/index.njk` - crate API reference landing page
- `docs/pages/*.md` - public documentation pages
- `_includes/` - shared layouts, nav, footer, and hero visual
- `assets/site.tailwind.css` - shared styling source (generates `site.generated.css`)

The generated static output lives in `docs/`.
