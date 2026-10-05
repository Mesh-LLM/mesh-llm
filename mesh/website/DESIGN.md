---
name: Mesh LLM Website
description: Developer infrastructure landing site for distributed local inference networking
colors:
  bg-deep: "#09090b"
  surface: "#0c0c0f"
  surface-2: "#111114"
  border: "#27272a"
  text-primary: "#f4f4f5"
  muted: "#a1a1aa"
  subtle: "#71717a"
  electric-blue: "#3b82f6"
  live-green: "#22c55e"
  amber: "#f59e0b"
  terminal-bg: "#050506"
  light-bg: "#fbfaf7"
  light-surface: "#ffffff"
typography:
  display:
    fontFamily: "Geist Sans, ui-sans-serif, system-ui, sans-serif"
    fontSize: "clamp(42px, 6vw, 76px)"
    fontWeight: 700
    lineHeight: 0.98
    letterSpacing: 0
  headline:
    fontFamily: "Geist Sans, ui-sans-serif, system-ui, sans-serif"
    fontSize: "clamp(30px, 4vw, 46px)"
    fontWeight: 700
    lineHeight: 1.05
    letterSpacing: 0
  body:
    fontFamily: "ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, sans-serif"
    fontSize: "19px"
    fontWeight: 400
    lineHeight: 1.5
  label:
    fontFamily: "ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace"
    fontSize: "12px"
    fontWeight: 600
    letterSpacing: 0
rounded:
  md: "8px"
  full: "999px"
spacing:
  section: "72px"
  card-padding: "18px"
  grid-gap: "14px"
components:
  button-primary:
    backgroundColor: "{colors.text-primary}"
    textColor: "{colors.bg-deep}"
    rounded: "{rounded.md}"
    padding: "0 14px"
    height: "38px"
  button-ghost:
    backgroundColor: "transparent"
    textColor: "{colors.muted}"
    rounded: "{rounded.md}"
    padding: "0 14px"
    height: "38px"
  card:
    backgroundColor: "{colors.surface}"
    textColor: "{colors.text-primary}"
    rounded: "{rounded.md}"
    padding: "{spacing.card-padding}"
---

# Design System: Mesh LLM Website

## 1. Overview

**Creative North Star: "Mesh Architecture"** — distributed systems visualized with engineering precision.

This is a developer infrastructure surface for an MIT-licensed tool that pools machines into a distributed inference network. The design reads like something you'd find in your terminal — capable, precise, built by engineers who ship real systems. Every pixel serves the product: animated topology diagrams demonstrate mesh networking, terminal mockups show actual commands, and the model catalog presents live data from Hugging Face.

The system explicitly rejects SaaS cream-beige marketing aesthetics, playful startup illustrations, gradient text, glassmorphism cards, and corporate cloud-provider sales pages. This is infrastructure tooling — it should feel like a well-engineered CLI, not a pitch deck. The dark-first theme respects developer environments: terminals, code editors, late-night debugging sessions.

**Key Characteristics:**
- Deep dark surfaces with structural shadow hierarchy (mechanical layer separation)
- Functional color palette: one accent blue for primary actions, green/amber for status only
- Tight typographic scale using Geist Sans with clamp-based fluid sizing
- Minimal radius (8px) — edges are gently curved, never soft
- Monospace treatment for all technical content (commands, flags, values)

## 2. Colors

A restrained palette where blue is the sole interactive accent and green/amber signal status exclusively. The neutral stack provides four levels of depth on dark surfaces.

### Primary
- **Electric Blue** (#3b82f6): Primary actions, brand identity mark, active states, section kickers. Appears as a soft background (rgba 12% opacity) for badges and eyebrow labels. Used on ≤10% of any given screen — its rarity is the point.

### Secondary
- **Live Green** (#22c55e): Status indicators only — live node dots, success states, active peer signals. Never used as a button or interactive element color.

### Tertiary
- **Amber** (#f59e0b): Warning and attention badges. Model status flags in the catalog table. Soft background variant (rgba 12%) for badge surfaces.

### Neutral
- **Deep Void** (#09090b): Page background. The canvas everything sits on. A near-black that isn't black — reduces harsh contrast while maintaining deep atmosphere.
- **Surface** (#0c0c0f): Card and container backgrounds. One step above void, providing the first structural layer.
- **Surface 2** (#111114): Elevated surfaces, badge backgrounds, ghost button states. The highest neutral surface.
- **Border** (#27272a): Structural dividers between cards, sections, and table rows. Used consistently at 1px.
- **Text Primary** (#f4f4f5): Headlines, body text, strong emphasis in tables. Near-white with warmth.
- **Muted** (#a1a1aa): Body copy, card descriptions, secondary navigation links. Readable without competing with primary content.
- **Subtle** (#71717a): Table headers, footer text, timestamps, and inactive states.

### Named Rules
**The One Voice Rule.** Electric Blue is the only interactive accent color. It marks primary buttons, active nav items, brand elements, and section kickers. Every other interaction uses neutral surfaces with text weight changes. Green and amber exist solely as status signals — never as action colors.

## 3. Typography

**Display Font:** Geist Sans (with ui-sans-serif, system-ui fallback)
**Body Font:** System sans-serif stack (ui-sans-serif, -apple-system, BlinkMacSystemFont)
**Label/Mono Font:** System monospace stack (SF Mono, Menlo, Monaco, Consolas)

**Character:** Tight, heavy headlines create architectural weight — the type feels like structural steel. Fluid clamp scales ensure headlines breathe on desktop while collapsing gracefully to mobile. Monospace appears wherever technical content lives: commands, flags, values, and model identifiers.

### Hierarchy
- **Display** (700, clamp(42px, 6vw, 76px), 0.98): Hero headline only. Maximum visual impact with tight line-height for the "one big statement" hero pattern.
- **Page Title** (700, clamp(38px, 6vw, 68px), 1.0): Inner page heroes (catalog landing, docs index). Slightly smaller than display to signal hierarchy.
- **Headline** (700, clamp(30px, 4vw, 46px), 1.05): Section titles on the homepage and feature cards.
- **Doc Title** (700, clamp(34px, 5vw, 58px), 1.05): Documentation page headers.
- **Body Lead** (400, 19px, 1.5): Hero description text and section copy. Maximum width: 650px.
- **Body** (400, 14–17px, 1.5): Card descriptions, documentation body text, table cells. Color: muted (#a1a1aa).
- **Label** (600, 12–13px, uppercase for headers): Section kickers, badge text, table column headers. Monospace for technical labels.

### Named Rules
**The Tight Headline Rule.** All display and headline type uses line-height ≤ 1.05 with letter-spacing: 0. The type should feel compressed and structural — like it's holding up the layout, not decorating it.

## 4. Elevation

Shadows serve a structural purpose — they separate content layers mechanically rather than creating atmospheric depth. Every shadow has a specific role in the layer hierarchy: terminal mockups float above cards, which sit on surface backgrounds, all resting on deep void.

### Shadow Vocabulary
- **Terminal Float** (`0 24px 60px rgba(0,0,0,0.35)`): Applied to terminal/code containers. The deepest shadow in the system — signals this is elevated content, not a flat card. Large blur radius creates mechanical separation from the background.
- **Card Default**: Cards use no shadow by default — they rely on `--surface` background + 1px border for layer definition. Shadows appear only when cards are elevated above their container.

### Named Rules
**The Structural Shadow Rule.** Shadows exist to separate layers, not create atmosphere. If a surface has a shadow, it should be because the content sits physically above another layer in the hierarchy — terminal mockups float over cards, sticky navs float over scrollable content. Flat surfaces use tonal contrast (surface vs background) and borders for definition.

## 5. Components

### Buttons
- **Shape:** Gently curved edges (8px radius)
- **Primary:** White-on-dark fill (#fafafa text on #09090b bg). Bold weight, uppercase-style compact sizing. Hover: shifts to zinc tone (#e4e4e7 background, #3f3f46 border). Minimum height: 38px.
- **Ghost:** Transparent background with 1px border (#27272a). Muted text color. Used for secondary actions and navigation links styled as buttons.
- **Hover / Focus:** Background shift only — no shadow elevation, no scale transforms. Focus-visible adds a 2px outline in blue (#60a5fa at 80% opacity) with 2px offset.

### Badges / Chips
- **Shape:** Fully rounded pills (999px radius)
- **Default:** Surface-2 background (#111114), muted text, 1px border. Minimum height: 24px, compact padding (0 × 8px).
- **Blue variant:** Blue-soft background (rgba at 12%), blue-tinted text (#bfdbfe), blue-tinted border (rgba at 25%). Used for technology tags and feature labels.
- **Green variant:** Green-soft background, green-tinted text (#bbf7d0). Status badges only.
- **Amber variant:** Amber-soft background, amber-tinted text (#fde68a). Warning/attention badges.

### Cards / Containers
- **Corner Style:** 8px radius throughout
- **Background:** Surface (#0c0c0f) with 1px border (#27272a)
- **Shadow Strategy:** No shadow at rest. Reference Elevation section for elevated states.
- **Internal Padding:** 18px on all sides
- **Grid Layout:** 3-column grid (homepage features), 2-column grid (docs cards), 14px gap

### Terminal / Code Blocks
- **Style:** Fixed dark background (#050506) regardless of theme. 1px border, 8px radius. Deep terminal float shadow (24px offset, 60px blur). Title bar with window chrome dots and subtle bottom border separator.
- **Syntax Highlighting:** Custom color scheme — commands in green (#86efac), flags in blue (#93c5fd), values in yellow (#facc15), comments in zinc (#71717a).
- **Copy Button:** Absolute-positioned, top-right corner. Semi-transparent dark background with white border at 16% opacity. Compact sizing (28px min-height).

### Navigation
- **Style:** Sticky header (64px height) with backdrop blur (16px), semi-transparent background matching page theme. Bottom border separator. Brand mark (blue SVG icon + bold text) on left, muted links on right.
- **Active State:** Foreground color shift to primary text weight 600 with left border accent.
- **Mobile:** Collapses to hamburger menu at 900px breakpoint, single-column layouts below 640px.

### Eyebrow Labels
- **Style:** Pill-shaped container (999px radius) with blue-soft background and blue-tinted border. Contains a green status dot + label text. Font: 12px, weight 500. Used above hero headlines to signal version, status, or category.

### Hero Topology Diagram
- **Style:** Animated SVG mesh network rendered behind hero content (z-index: 0). Dashed connection lines with opacity pulse animation. Glowing packet dots traverse node-to-node paths on staggered timelines. Concentric ring animations radiate from node cores. Labels appear/disappear with a pop-in animation cycle.

## 6. Do's and Don'ts

### Do:
- **Do** use Electric Blue (#3b82f6) as the sole interactive accent — primary buttons, active nav, brand mark, section kickers.
- **Do** keep headlines tight: line-height ≤ 1.05, letter-spacing: 0, weight 700. The type should feel structural.
- **Do** use monospace for all technical content: commands, flags, values, model names, terminal output.
- **Do** apply the Terminal Float shadow (24px offset, 60px blur) to code containers — this is the deepest elevation in the system.
- **Do** respect the surface hierarchy: bg-deep → surface → surface-2. Each step up signals one layer of structural elevation.
- **Do** constrain body copy to 650px max-width for readability.
- **Do** use badges (green/blue/amber) for status and categorization only — never as interactive elements.

### Don't:
- **Don't** use gradient text, glassmorphism cards, or side-stripe borders. These are SaaS marketing clichés that belong to the anti-references this system explicitly rejects.
- **Don't** use green (#22c55e) or amber (#f59e0b) as button colors or interactive accents. They signal status — live nodes, warnings, model availability. Nothing else.
- **Don't** apply shadows to flat cards. Cards define their layer through surface color + border. Shadows are reserved for elevated containers (terminals, sticky navs).
- **Don't** exceed 8px radius on standard components. The system uses gently curved edges — not soft, rounded startup aesthetics.
- **Don't** use playful illustrations or decorative graphics. If it doesn't demonstrate architecture, show data, or display a command, it doesn't belong.
- **Don't** create cream-beige backgrounds or warm neutral palettes for light mode. The light theme uses precise off-whites (#fbfaf7) with deliberate border tones (#ddd8cc).
