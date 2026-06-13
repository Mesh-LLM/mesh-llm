# AGENTS.md - Mesh LLM Website

Project-specific instructions and references for AI agents working on the Mesh LLM website.

## Project Overview

- **Static Site Generator**: Eleventy v3.1.2 with Nunjucks templates
- **Build Command**: `npm run build` (production), `npm run dev` (development server)
- **Design System**: Dark-first CSS custom properties in `src/assets/site.css`
- **Typography**: Geist Sans font family

## Build and Deployment Contract

- `website/` is the maintained source tree for the public static site.
- `docs/` at the mesh-llm repo root is mixed by path: generated website output lives alongside authored project Markdown docs.
- Generated website artifact paths are `../docs/index.html`, `../docs/CNAME`, `../docs/install.sh`, `../docs/install.ps1`, `../docs/mesh-llm-logo.svg`, `../docs/funding.json`, `../docs/.well-known/`, `../docs/assets/`, `../docs/catalog/`, `../docs/docs/`, and `../docs/pagefind/`.
- Authored project docs such as `../docs/MESHES.md`, `../docs/design/**`, `../docs/plugins/**`, and `../docs/specs/**` remain source and should be edited in place.
- Do not hand-edit generated website artifact paths; edit `src/**` and rebuild.
- Production build order is Tailwind CSS -> Eleventy -> Pagefind.
- `npm run build` writes Tailwind output to `src/assets/site.generated.css`, runs Eleventy with output `../docs`, then indexes `../docs` with Pagefind.
- From the mesh-llm repo root, use `just website-build` for production output, `just website-dev` for the local dev server, and `just website-clean` to remove generated website output while preserving authored docs.
- Eleventy passthrough copies `src/CNAME`, `src/funding.json`, `src/.well-known/`, `src/assets/`, `src/mesh-llm-logo.svg`, and repo-root `../install.sh` / `../install.ps1` into the generated `docs/` tree.
- `src/assets/site.generated.css`, `node_modules/`, Eleventy caches, and browser/test artifacts are generated/local artifacts, not source.

## Design System Reference

### Impeccable Skills & Design Tokens

The `.impeccable/design.json` file contains the authoritative design system:

```json
{
  "extensions": {
    "colorMeta": { ... },      /* Color palette with tonal ramps */
    "typographyMeta": { ... },   /* Typography scale and purposes */
    "shadows": [ ... ],          /* Shadow tokens */
    "motion": [ ... ],           /* Animation tokens */
    "breakpoints": [ ... ]       /* Responsive breakpoints */
  },
  "components": [ ... ],         /* Component definitions with HTML/CSS */
  "narrative": { ... }           /* Design philosophy and rules */
}
```

**Key References**:
- **Colors**: Deep Void (#09090b), Surface (#0c0c0f), Electric Blue (#3b82f6), Live Green (#22c55e)
- **Typography**: Display (hero headlines), Headline (section titles), Body Lead (max-width: 650px), Label/Mono (technical content)
- **Shadows**: Terminal Float (deepest elevation, for code containers)
- **Animations**: Mesh Line Pulse, Packet Traverse, Node Pulse

### Design Principles

1. **The One Voice Rule**: Electric Blue is the only interactive accent color
2. **Tight Headline Rule**: Display/headline type uses line-height ≤ 1.05 with letter-spacing: 0
3. **Structural Shadow Rule**: Shadows separate layers mechanically, not atmospherically

## Animation Guidelines

### Required Library: Anime.js v4

**ALWAYS use [Anime.js](https://animejs.com/documentation/) for animations.** This is the official animation library for this project.

#### Installation & Setup
```bash
npm install animejs@4.x
```

#### Basic Usage Examples

```javascript
// Simple property animation
import { animate } from 'animejs';

animate('.element', {
  opacity: [0, 1],
  translateY: ['20px', '0'],
  duration: 0.5,
  easing: 'easeOutQuad'
});

// Timeline for complex sequences
import { timeline } from 'animejs';

const tl = timeline({ autoplay: false });
tl.add(animate('.hero-title', { opacity: [0, 1], duration: 0.6 }), 0);
tl.add(animate('.hero-subtitle', { opacity: [0, 1], delay: 0.2 }));
tl.add(animate('.cta-button', { scale: [0.95, 1] }, '-=0.3'));

// Scroll-triggered animations
import { onScroll } from 'animejs';

onScroll({
  target: '.animated-section',
  threshold: 0.2,
  onEnter: () => animate('.section-content', { opacity: [0, 1], translateY: ['30px', '0'] })
});
```

#### Key Features to Leverage
- **Timeline**: For complex sequential animations with precise timing control
- **ScrollObserver (`onScroll`)**: For scroll-triggered animations without external libraries
- **Staggering**: Built-in stagger utilities for list/grid animations
- **Easings**: Rich easing functions including springs and cubic beziers

#### Animation Patterns to Use
1. **Hero Animations**: Subtle entrance animations (fade + slight translate)
2. **Scroll Reveals**: Elements animate in as they enter viewport
3. **Hover States**: Micro-interactions on buttons/cards with spring easings
4. **Page Transitions**: Smooth transitions between sections

#### Animation Patterns to Avoid
- No gratuitous bouncing or excessive motion
- Keep animations purposeful and performance-conscious
- Respect `prefers-reduced-motion` media query
- Maintain the technical, infrastructure aesthetic - no playful startup animations

## Development Conventions

### File Structure
```text
src/
├── assets/          /* CSS, images, fonts */
│   ├── site.css     /* Core authored CSS/design tokens */
│   ├── site.tailwind.css     /* Tailwind input */
│   └── site.generated.css    /* Generated by npm run build:css */
├── _data/           /* Global data files */
├── _includes/       /* Layouts and partial templates */
├── catalog/         /* Catalog viewer */
├── docs/            /* Public documentation pages */
└── index.njk        /* Homepage template */
```

### CSS Architecture
- Use CSS custom properties for theming
- Follow the dark-first color palette defined in `design.json`
- Maintain structural shadow hierarchy (no decorative shadows)
- Keep component styles consistent with design system tokens

### Template Conventions
- Nunjucks templates use `.njk` extension
- Extend base layouts using `{% extends "layouts/base.njk" %}`
- Use includes for reusable components: `{% include "partials/component.njk" %}`
- Data files in `_data/` are automatically available to all templates

## Key URLs & Resources

- **Anime.js Docs**: https://animejs.com/documentation/
- **Design System**: `.impeccable/design.json` (authoritative tokens)
- **Product Vision**: `PRODUCT.md` and `DESIGN.md` at project root
