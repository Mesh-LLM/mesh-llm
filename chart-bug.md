# Recharts "Maximum Update Depth Exceeded" Bug — Investigation Notes

> **Status (2026-08-17): the bug is LIVE and UNFIXED on this tree.** The `/logs` page crashes
> with `Maximum update depth exceeded` (route error boundary: "Render fault" / "Something went
> wrong"). The fix layers described in the original report were verified to exist **nowhere**
> in the repository. This file was cleaned of stale claims on 2026-08-17; it now records the
> confirmed root cause, the fix design (unapplied), and the test coverage guarding the fix.

## Environment (verified)

- `crates/mesh-llm-ui`: react/react-dom **19.2.8**, recharts **^3.10.1**, react-redux v9
  (synchronous `defaultNoopBatch`), @playwright/test 1.62.1, vitest 4.1.10, TypeScript 5.9.3.
- Chart component: `src/features/logs/components/EventsOverTimeChart.tsx`, rendered by
  `LogsLedger.tsx` as `<EventsOverTimeChart now={selectedLedgerRange.endMs} ...>` — `now` is
  always provided in normal UI flow, so the chart's internal advancing clock
  (`useAdvancingChartClock`) runs **disabled**; it only activates when `now` is `undefined`.
- `chart.tsx` (`src/components/ui/chart.tsx`) hosts the `ChartContainer`/`ChartTooltip`
  primitives used by the chart.

## Symptom

`/logs` renders the route error boundary instead of the chart:

- Alert "Render fault", heading "Something went wrong", paragraph "This section failed to
  render, but the rest of the app can stay available."
- Error: `Maximum update depth exceeded. This can happen when a component repeatedly calls
  setState inside componentWillUpdate or componentDidUpdate. React limits the number of nested
  updates to prevent infinite loops.` — boundary scope "Route section".

## Root Cause (stack-confirmed)

recharts v3.10.1's internal redux store + react-redux v9's synchronous `defaultNoopBatch`
notification. On measurement/data-update cycles the following dispatch chain repeats until
React's 50-depth limit:

```
ResizeObserver → SizeDetector → setContainerSize → ResponsiveContainerContext
  → <CategoricalChart> re-render → new props object
  → useEffect([props]) → dispatch(updateOptions(props))
  → redux store notify → forceStoreRerender (sync)
  → subscribers re-render → new props → repeat → 50 depth limit → CRASH
```

Live stack trace (captured from the error-context artifact of the e2e run):
`forceStoreRerender (react-dom_client.js)` ← `Object.callback (recharts.js:8187)` ←
`defaultNoopBatch (recharts.js:8173)` ← `Object.notify (recharts.js:8184)` ←
`notifyNestedSubs (recharts.js:8243)` ← `handleChangeWrapper (recharts.js:8246)` ←
`wrappedListener (recharts.js:6195)`.

Feeding mechanisms in recharts v3.10.1:

- `ReportChartProps` is unmemoized; `useEffect([dispatch, props])` with `props` a new object
  literal on every `<CartesianChart>` render.
- `SetXAxisSettings`/`SetYAxisSettings`: `restProps = _objectWithoutProperties(props)` creates
  a new object each render → `useMemo([restProps])` recomputes → `useLayoutEffect` dispatches
  `replaceXAxis`/`replaceYAxis`.
- Other dispatch sites: `setChartData`, `updateXAxisHeight`, `updateYAxisWidth`.

## Trigger conditions (updated)

The original report claimed the loop only fired under Playwright `clock.setFixedTime()`
(which patches only `Date.now()` while real `setTimeout`/`setInterval`/`ResizeObserver` keep
firing; `clock.install()` pauses timers and hides the loop). That framing is **incomplete**:
with correct live-mode mocks the crash reproduces **without any clock manipulation** — the
real-timer variants (idle, resize-storm, stream-burst, row-click) crash too. The essential
ingredients are:

1. The chart mounts with **real data from live mode** (`DATA_MODE_STORAGE_KEY` =
   `'mesh-llm-ui-preview:data-mode:v2'` set to `'live'`). Dev builds default to **harness
   mode**, which serves built-in fixtures and ignores `/api/logs/*` mocks — harness mode
   never kicks the loop.
2. The live-stream machinery perturbs state after mount (e.g. SSE `stream_error` →
   `hydrateAuthoritatively` refetch → new array instances → chart re-render).

## Fix design — NOT APPLIED (verified absent)

The original investigation designed three fix layers. They were verified to exist **nowhere**
in the repository (not HEAD, not worktree, not reflog, not stash). Documented here as the
blueprint for the fix:

1. **Stabilize inline props on recharts components** — `EventsOverTimeChart.tsx`: hoist
   `tickFormatter` to a module-level constant (currently an inline arrow), hoist the inline
   `tick={{ fill, fontSize }}` literals, and the conditional inline `cursor` object on
   `<ChartTooltip>`. recharts' `axisPropsAreEqual` uses strict equality for non-allowlisted
   props, so inline objects defeat its memoization.
2. **Memoize `<ChartTooltip>`** — `chart.tsx`: wrap in `React.memo` with a custom comparator
   (`chartTooltipPropsAreEqual`) that uses `shallowEqual` for the `cursor` prop (in recharts'
   shallow-compare allowlist) and reference equality otherwise. Prevents the tooltip's
   `useEffect([dispatch, props])` from re-firing on every parent render.
3. **Bypass `<ResponsiveContainer>` entirely** (most robust; recommended by the librarian
   research) — `chart.tsx` still renders bare `RechartsPrimitive.ResponsiveContainer`.
   Replace it with a custom `ResizeObserver` that measures the parent once on mount and passes
   explicit numeric `width`/`height` to the chart child via `React.cloneElement`. This
   eliminates recharts' internal measurement dispatch chain, the primary loop driver.

Layers 1–2 are defense-in-depth against the sync notify chain; layer 3 removes the
timing-dependent ResizeObserver→dispatch chain. All changes are presentation-layer only; no
business logic is involved.

## Test coverage (current state)

**`crates/mesh-llm-ui/e2e/logs/logs-chart-stability.spec.ts`** — the permanent regression
suite, following the tracked `log-workflows` mock conventions (`**/api/logs/**` routes,
live-mode `addInitScript`, valid UUIDs via `padStart(12)`). Seven variants:

1. frozen time (`setFixedTime`) — renders real data without exceeding React update depth
2. frozen time — high-volume dataset (370 rows; ledger window capped at
   `LOG_EVENT_WINDOW_LIMIT = 64`, so the gate asserts the legend reports `Requests64` plus
   bars > 0 and no depth errors)
3. frozen time — chart tracks viewport growth (`wLarge > wSmall * 1.15`, guards a
   fixed-width "measure once" wrapper regression)
4. real timers — idle static mount
5. real timers — resize-storm (repeated viewport resizes)
6. real timers — stream-burst (SSE audit events mutating chart data)
7. real timers — row-click (opening the request inspector shifts the chart layout)

**The suite is intentionally red until the fix lands**: in live data mode every variant hits
the crash. It flips green once the fix above is implemented, then guards each perturbation
shape against regressions.

**`src/features/logs/components/EventsOverTimeChart.render-loop.test.tsx`** — Vitest guard
(fake timers + synchronous ResizeObserver stub, fresh props arrays across parent re-renders).
Green on both pre-fix and post-fix code (jsdom cannot reproduce the loop); complementary,
non-discriminating. Included in the 1403 passing unit tests.

**Tracked `e2e/logs/log-workflows.spec.ts`** — red on this tree (11/13 standalone) for the
same crash; the 2 passing tests use `streamMode: 'unavailable'` (aborted stream → no live
perturbation). Full E2E smoke (via `just test-all` step 11): 23 failures, all on the `/logs`
surface (log-workflows 13, logs-chart-stability 6, request-inspector 3, logs-a11y,
schema-controls) — same crash.

### Why the original repro looked green (silent gaps, fixed in the permanent suite)

1. **Harness-mode fixtures**: the Playwright webServer runs `vite` (dev build);
   `DataModeProvider` defaults to harness mode, which serves built-in fixtures and ignores
   `/api/logs/*` mocks unless the spec opts into live mode via `addInitScript` setting
   `DATA_MODE_STORAGE_KEY` to `'live'`. The original repro specs never did; the tracked specs
   do.
2. **Invalid mock request IDs**: `requestRow` built `requestId` with an 8-character final
   UUID group (`padStart(8)`), which fails `requestIdSchema`; the whole requests page then
   failed to parse and was silently dropped (only audit rows rendered). Fixed with
   `padStart(12)`, matching the tracked `REQUEST_ID` shape
   `00000000-0000-4000-8000-000000000001`.

## Test inventory (2026-08-17) — what the unstaged tests were for

Every untracked test that existed in the tree when the inventory was written, what it was
added for, and its disposition. The probe/repro specs were scratch work from the original
investigation and re-verification sessions; none were ever committed except as folded into
the permanent suite.

| Test file (as found) | Added for | Disposition |
|---|---|---|
| `e2e/logs/repro-loop-faithful.spec.ts` | "Faithful" Playwright repro of the render loop under `clock.setFixedTime` (the documented trigger), with full console/pageerror capture | **Refactored** → its dataset/assertions are the core of `e2e/logs/logs-chart-stability.spec.ts` (frozen-time variant) |
| `e2e/logs/trigger-isolation.spec.ts` | Real-timer (no `page.clock`) trigger-isolation matrix: idle / row-click / resize-storm / stream-burst, to find which perturbation triggers the crash | **Refactored** → its 4 variants became the `real timers (no clock)` tests in `logs-chart-stability.spec.ts` |
| `e2e/logs/stress-resize-loop.spec.ts` | Resize-storm stress + unique assertion that the chart tracks viewport growth (`wLarge > wSmall * 1.15`, guards a fixed-width "measure once" wrapper regression) | **Refactored** → viewport-growth assertion became the `frozen time` "chart tracks viewport growth" test |
| `e2e/logs/mount-check-probe.spec.ts` | Fidelity probe: 370 request rows under frozen time, chart must render real bars (fidelity gate `bars >= 150`) | **Refactored** → high-volume variant in `logs-chart-stability.spec.ts`; the `>= 150` bar gate was **recalibrated** (see notes) |
| `e2e/logs/clock-probe.scratch.spec.ts` | Scratch: probe rAF cadence / `Date.now` pinning / real-timer wall time / ResizeObserver under `setFixedTime` | **Deleted** — no assertions, pure diagnostics |
| `e2e/logs/debug-ab.spec.ts` | A/B probe: with mocked live data, poll for the error boundary (`BOUNDARY`) vs a healthy heading (`HEADING_VISIBLE`) | **Deleted** — superseded by the permanent spec; messy poll/then/catch pattern |
| `e2e/logs/diag-dispatch.spec.ts` | Diagnostic of recharts redux dispatch, requires a `node_modules` patch in `RechartsStoreProvider.js` gated on `window.__rcDiag` | **Deleted** — depends on a node_modules patch, not a viable permanent test |
| `e2e/logs/probe-row-click.spec.ts` | Diagnostic: click a row and dump chart/console state over time | **Deleted** — no assertions; mocked wrong (pre-refactor) API shapes |
| `e2e/logs/repro-faithful.spec.ts` | Faithful replica of `log-workflows.spec.ts` test #1 (lifecycle → inspector → artifacts) with full console capture | **Deleted** — duplicates the tracked `log-workflows` coverage; console-capture value is covered by the new spec's `captureErrors` |
| `e2e/logs/repro-burst-feed.spec.ts` | Temporary probe: high-volume live SSE feed under frozen time (400 events) | **Deleted** — wrong SSE/API shapes and a wrong `baseURL` (port 8765, dev server, not the Playwright webServer port) |
| `e2e/logs/repro-resize-feed.spec.ts` | Temporary probe: resize storm + repeated reloads under frozen time | **Deleted** — wrong API shapes and port; superseded by the resize-storm variant |
| `src/features/logs/components/EventsOverTimeChart.render-loop.test.tsx` | Vitest regression test (fake timers + synchronous ResizeObserver stub, fresh props arrays across parent re-renders) | **Kept** as a green guard — passes on both pre-fix and post-fix code (jsdom cannot reproduce the loop); complementary to the Playwright suite |

Notes:

- The high-volume `>= 150` bar gate from `mount-check-probe` was dropped: under the 64-row
  `LOG_EVENT_WINDOW_LIMIT` cap and the default 5-minute bucket interval, that many bars are
  impossible by design. The permanent variant asserts legend `Requests64` instead (proves the
  dataset reached the ledger and was windowed, not silently dropped).
- All 12 files above were either deleted or folded into `logs-chart-stability.spec.ts`;
  the permanent suite + the vitest guard are the surviving artifacts.

## Recommended next steps (for the fixer)

1. Implement the three fix layers (see "Fix design — NOT APPLIED"), in order of impact:
   layer 3 (ResponsiveContainer bypass) first, then 1 and 2 as defense-in-depth.
2. Re-run `logs-chart-stability.spec.ts` (expect all 7 variants green) and the tracked
   `log-workflows.spec.ts` (expect recovery from 11/13 failures).
3. Re-run vitest + typecheck + build; the render-loop vitest guard must stay green.