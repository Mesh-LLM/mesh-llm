# Product

## Register

brand

## Users

Broad developer spectrum: individual enthusiasts pooling personal machines (Mac Pro + gaming rig), ML engineers and small teams running distributed inference across GPU clusters without cloud dependencies, and AI tool builders needing reliable local model backends. They share a need for powerful models that exceed single-machine capacity, a preference for self-hosted infrastructure, and familiarity with the local LLM ecosystem (llama.cpp, Ollama, GGUF).

## Product Purpose

Mesh pools machines into a distributed inference network so large local models can be served across multiple devices through one OpenAI-compatible endpoint. It handles discovery, placement, routing, and model serving — making multi-machine inference as simple as setting `OPENAI_BASE_URL`. Success means developers run frontier-quality models on hardware they already own, without cloud costs or vendor lock-in.

## Brand Personality

Technical, confident, open-source. The voice is that of a capable engineer who ships real infrastructure. No marketing fluff — specific numbers, concrete architecture, honest about trade-offs. The site demonstrates what Mesh does through live topology diagrams, terminal output, and a browsable model catalog rather than abstract claims. Community ethos: MIT-licensed, peer-to-peer, no central directory.

## Anti-references

SaaS cream-beige marketing sites with gradient text and glassmorphism cards. Playful startup aesthetics (illustrations, rounded corners everywhere). Cloud provider pages that feel corporate and salesy. This is developer infrastructure — it should read like a tool you'd find in the terminal, not a pitch deck.

## Design Principles

1. **Show, don't tell** — Demonstrate architecture through diagrams, live data, and real terminal output rather than feature lists.
2. **Engineer-to-engineer** — Write for people who understand the problem space; no hand-holding, no buzzwords.
3. **Practice what you preach** — The site itself is distributed, open-source, self-hosted. The medium matches the message.
4. **Dark-first developer surface** — Respect the environments where users actually work: terminals, code editors, dark mode.

## Accessibility & Inclusion

WCAG AA target. Dark theme with verified contrast ratios (body text ≥ 4.5:1). Reduced-motion support for animated topology diagrams. Monospace font fallbacks for screen readers.
