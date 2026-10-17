# What is copied from the mesh-llm console

The Evidence page is adapted from the mesh-llm console: it began as a
`/capsules` tab (`crates/mesh-llm-ui/src/features/capsules/**`) in a copy of
the console built on upstream
[Mesh-LLM/mesh-llm](https://github.com/Mesh-LLM/mesh-llm) main at commit
`d635e6156c87a2e7e7d0c0c7173a2b388fb225b2` (2026-09-28), the upstream
console it was last synced against. The shared UI pieces (`src/components/ui`,
`src/lib`) come from that console. The source layout is kept (`@/` → `src/`).
mesh-llm is Apache-2.0, as is this repository.

At the last sync, each file that had drifted was merged three ways, keeping
this page's changes below where the two differ. Files that tab had and this
page does not, and why:

| Console tab file | Here |
| --- | --- |
| `lib/settlement-view.ts` (+ test), `components/SettlementRow.tsx` | **ported**, unchanged: the row's payment chips and entries, the Peers payments line and Integrity's Close card. Wording gate: `lib/settlement-copy.test.ts` |
| `lib/their-history-view.ts` (+ test), `components/PeerHistoryTab.tsx` (+ test) | **ported**, unchanged: the peer drill's "Their log, as shown to you" tab |
| `lib/ask-for-record.ts` (+ test) | **ported**, unchanged |
| `api/evidenceRequestClient.ts` (+ test) | **ported**: the ask goes to this plugin's `tools/mesh_evidence_request` (the console's host route forwarded to the same tool), and the announced key comes from this plugin's `http/peer-key` route instead of the host's reply |
| `components/PeerInspector.test.tsx`, `components/SeeInLogsLink.test.tsx` | **ported**; the see-in-Logs test pins the plain-text fallback below |
| `components/StopRoutingDialog.tsx`, `components/PeerRoutingSection.tsx` (+ test), `api/peerBlocksClient.ts`, `api/usePeerBlocks.ts` | **not ported**: stopping routing needs the host's local block list, which a plugin page can't reach yet. The drill says in plain text that this page can't stop routing (`ROUTING_NOT_ON_THIS_PAGE`), and the Peers row shows no "routing stopped" badge |
| `components/ChatWithNodeButton.tsx` (+ test), `api/routeTargetClient.ts` (+ test), `api/useRouteTarget.ts` | **not ported**: pointing Chat at one node needs the host's `/api/route-target`. No button and no "Your chats go here" badge |
| `lib/chat-evidence-link.ts` (+ test) | **not ported**: it serves the console's own Chat and Logs pages, not this page |
| `pages/AccountabilityPage.tsx`, `pages/CapsulesExchangeRedirectPage.tsx` (+ test), `__fixtures__/.gitignore` | **not ported**: console router glue |


## The page (was the console tab), `src/features/capsules/**`

Copied as-is except for these files:

| File | Change |
| --- | --- |
| `api/client.ts` | ledger reads go to this plugin's routes through `host.network.fetchPlugin` (JSON-wrapped); same exports and return shapes |
| `api/sidecarClient.ts` | pane reads go to this plugin's `panes/*` routes; `PaneFetchError` is the host seam's `PluginRouteError` |
| `api/peerLedgerFetchClient.ts` (+ test) | the tool route is plugin-relative (`tools/mesh_ledger_fetch`), so it now reaches `capsules`; the console tab addressed the retired name `admission-policy` |
| `pages/LedgerPage.tsx` | router `navigate` → `navigateHost`; the pane-error copy names the plugin (503) and adds the 404 "this plugin build doesn't serve its evidence panes yet" case |
| `components/ExchangeIdCell.tsx` | router `navigate` → `navigateHost('/logs?focusExchangeId=…')` |
| `lib/peer-mesh-status.ts` | calls the peers-only status adapter (below) |
| `lib/ledger-logs-non-overlap.test.ts` | the console's Logs column keys are pinned (they are console code) instead of imported |
| `retired-copy.test.ts` | scans every shipped source of the bundle and the plugin's Rust panes |
| `pages/EvidencePage.tsx` | **new**: the providers the console app supplied (query client, data mode fixed to live, tooltip provider) + `?focusExchangeKey=` |
| `pages/AccountabilityPage.tsx`, `pages/CapsulesExchangeRedirectPage.tsx` (+ test) | **removed**: console router glue; the deep link is `?focusExchangeKey=` on the plugin page |
| `lib/peer-routing-view.ts` (+ test) | **ported later** from the console tab: `dealingsLines` only (the drill's "Your dealings with them"); the block-store half needs the host's local block list, which this page does not reach |
| `components/PeerInspector.tsx` | the Overview shows "Your dealings with them" (`dealingsLines`) and a routing section that says plainly this page can't stop routing, in place of the console's Stop routing button |
| `lib/tooltip-copy.ts` | adds `YOUR_DEALINGS_TITLE` and `WITNESS_OFF` (console copy) and `ROUTING_NOT_ON_THIS_PAGE` |
| `lib/your-records.ts` (+ test) | **ported later**: `heroStatusLine`, the `What you share` rows, the `Your records` facts and the `Clean up records` words, unchanged (the hero line says "no outside witness" instead of "checkable only by you"); the `Your prompts` pill is not ported |
| `components/YourRecordsDialog.tsx` | **ported later**, unchanged; the page passes the checkpoint's "no later than" time in local time |
| `components/CleanUpRecordsDialog.tsx` (+ test) | **ported later**, unchanged |
| `api/recordsClient.ts` | **ported later**: same types and calls; the tool route is plugin-relative through the host's `fetchPlugin` |
| `lib/use-your-records.ts` | **ported later**: the records-status query only (`useRecordsStatus`); the console's stored-text probe feeds a pill this page does not show |
| `lib/local-time.ts` | **ported later**, unchanged; row and drill times (`ExchangeStreamRow`, `PeerInspector`, `PeerExchangeInspector`) and the Integrity "no later than" line use it |
| `lib/entry-row-chips.ts`, `components/ExchangeRowChips.tsx` | the chips show the console's plain labels (`ENTRY_ROW_CHIP_LABEL`); the console's "covered" (◐) mark is not ported: this page keeps ✓ / ✗ / – and says per result, in `ENTRY_CHIP_RESULT_TOOLTIPS`, that the checkpoint chip is the node's count, not a checked proof |
| `components/ExchangeIdCell.tsx` `SeeInLogsLink` | the row's "see in Logs" is plain text (`SEE_IN_LOGS_NOT_ON_THIS_PAGE`): opening Logs at one exchange needs the host's Logs to carry the exchange id, which it doesn't yet |
| `components/ExchangeStreamRow.tsx` | the ask goes through the page (`onAskForRecord`), never on sample data; no Chat-with-this-node button; the console tab's `FETCHED_CLOSE_TEXT` is this page's `CLOSED_FROM_FETCH_NOT_SAVED` (same words), shown for an asked-for record and a fetched one; "your own copy fails its checks" shows beside any badge |
| `lib/integrity-view.ts` (setup steps) | the witness step has no "Turn on a witness ↗" link: the console's `/configuration/plugins` is a host route this page can't count on |
| `pages/LedgerPage.tsx` (ask, Close card) | "Ask them for their record" and the Close card as in the console; no routing or chat-target controls; the Peers section's unattributed count is read from the Exchanges rows, as in the console |
| `lib/peer-row-view.ts` | the Peers row's endpoint alias is cut to 16 characters, as in the console |
| `lib/integrity-view.ts` | plain words ahead of the console: "witnessed" (only when a witness holds a checkpoint) for "registered", local time, and the console's reworded owner-identity sentence |
| `pages/LedgerPage.tsx` | the hero line and the `Clean up records` button + dialog, as in the console |

## Console primitives the page imports

These are copied unchanged, so the page looks like the console: the
`src/components/ui/*` files, `src/lib/{cn,utils,env,vram,copyStateLabel,useClipboardCopy,format-model-size}.ts`,
`src/lib/api/*`, `src/lib/query/query-keys.ts`, `src/lib/data-mode/*`,
`src/lib/test/setup.ts`, `src/features/app-tabs/types.ts`,
`src/features/app-shell/lib/status-types.ts`, `src/features/chat/lib/chat-types.ts`,
and `src/features/network/api/*`, **except** `status-adapter.ts`, which is
cut to its peers-only subset. The page reads only `peers` from it, and the
dashboard half pulled in the console's dashboard fixtures.

The console styles these primitives with global component classes (e.g.
`surface-menu-panel`, `ui-control-ghost`) and theme tokens. Those classes
stay the console's: mounted in the console, the page uses them as they are.
Only Tailwind utilities are compiled into the page's own sheet
(`src/styles/evidence.css`).

## Re-syncing

When a console primitive changes upstream in a way the page should pick up,
copy the file again from mesh-llm-ui and re-run `pnpm typecheck && pnpm test`.
The page never imports console source at build time, so an upstream change
cannot break the installed bundle.
