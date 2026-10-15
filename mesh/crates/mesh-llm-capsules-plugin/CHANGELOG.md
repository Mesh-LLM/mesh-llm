# Changelog

## Unreleased

### Moved

- **The plugin is now developed in the mesh-llm repository,** as the `mesh-llm-capsules-plugin`
  crate under `mesh/crates/mesh-llm-capsules-plugin`. The executable, its id and its data
  directory are still `capsules`.
- **Its version follows mesh-llm's:** 0.1.4 is followed by mesh-llm's version (0.78.0 at the move),
  in `plugin.toml`, in what the plugin reports to its host, and in the `developer` field of every
  record it seals (`capsule-emit-mesh/<version>`).
- **Release builds keep their symbol table** (mesh-llm's release profile strips debug info only),
  which makes the executable a few MB larger.

### Changed

- **An evidence answer is now the interop profile's signed envelope. Upgrade both nodes.** The
  protocol crate is `evidence-request` `=0.2.0` (what `capsule-emit-evidence-request` became when it
  moved to agent-action-capsule), which signs and checks the envelope of agent-action-capsule's
  `spec/interop-profile-v0.md` §6.2 (`evidence-interop/0`), as the Go, Python and evidencebook
  implementations do. Before, the answer was a pre-profile `{artifact, material, envelope}` that
  no other implementation signed. A node on an older version cannot read the new answer, and this
  node cannot read the old one.
  - **What an anchor is.** A checkpoint is pinned by the SHA-256 of its COSE wire statement (the
    `checkpoint_cose` the checkpointer already writes beside each checkpoint), not by the digest of
    its JSON form. A checkpoint without its own statement anchors no answer.
  - **A record answer** is the record's ledger line, as before, with the profile's Checkpointed
    Local Log material for its `capsule_id` (the anchor's statement, the inclusion proof, and the
    consistency proof when the checkpoint extends another). The line is not the `capsule_id`
    preimage, so a profile verifier reports that check `NOT_CHECKED`, never `FAIL`; this node's
    requester recomputes the `capsule_id` and checks the proofs, and also accepts a record served as
    its own octets, which the profile's check passes. The line keeps the producer's signature, which
    the page's "Ask them for their record" checks. The page reads the artifact as the profile
    carries it, unpadded base64url.
  - **The other answers** take evidencebook's artifact forms (record sets, the checkpoint list,
    `history_card/1`) and `served_summary/1` the same shape. The profile has no material for them
    and no carrier for witness receipts, so this node also sends, where a profile verifier reads
    nothing: the anchor's statement and one inclusion proof per record for a record set, and the
    witness receipts of the checkpoints it carries in a `witness_receipts` member beside the
    envelope. The requester checks all of it as before.
  - The builder stays in the plugin (`evidence_artifact`): evidencebook's builder works over its
    own `CllSubstrate`, which would open the node store the checkpointer owns.
  - A `served_summary/1` answer's verification now reports `records: []`, like every other
    answer, not `0`.
- **`capsule-emit` 0.0.8 and `checkpointed-local-log` 0.3.1**, one CLL line in the build, and no
  `evidencebook`. capsule-emit 0.0.8 turns on `serde_json`'s `arbitrary_precision` for the whole
  binary; the strict parser for pushed bodies (`strict_json`) now reads numbers in that form, still
  refuses a float that overflows, and refuses a key that spells `serde_json`'s private number
  token.
- **The evidence-request parity run is judged by agent-action-capsule's vectors.** The
  self-generated corpus and golden answers are gone from `vectors/parity/evidence_request/`; the
  draft's conformance vectors are copied into `vectors/aac-evidence-request/` (tag
  `crates/evidence-request-v0.2.0`), and the responder and requester are run against them. The
  sharing-policy cases, which no vector states, keep their reviewed expected answers in the
  runner.
- **Who asks for evidence is now proven, not stated. Upgrade both nodes.** Each evidence request
  this node sends is signed with the key its host announces for it. The request carries
  `requester_auth` (`key_id`, `sig`), an `audience` (the node it is sent to), a `nonce` and a
  `deadline`.
  - **Before it looks anything up, a node answering checks:**
    - the signature;
    - that the request is addressed to it;
    - that the nonce is new;
    - that a host announced the key for the node the request comes from.
  - Records and Standard History are then answered only to that node. The id a request states is
    no longer trusted; if several nodes announced the same key, it picks between them.
  - **Refused:**
    - a request stating one node's id while signed by another;
    - a request addressed to another node, so a node cannot forward a request it received and
      read the answer;
    - a replayed nonce;
    - a deadline more than 5 minutes ahead.
  - **An unsigned request now gets a signed refusal wherever the answer depends on who asks:**
    `not_authorized` for a record, `policy_declined` for Standard History. It never gets
    silence.
  - **Two settings still answer unsigned requests:** `share_history_segments = peers` for records,
    and `share_standard_history = on` for Standard History. Under them, a valid signature under a
    key no host announced is answered as an unsigned request, and nothing more. Any other failed
    check is still refused.
  - **A node on 0.1.5 or older sends unsigned requests,** so after you upgrade, its requests for
    your records or Standard History are refused until it upgrades too.
  - **A node signs only once its host has taken its key.** On a host without `plugin_keys.v1` its
    requests go out unsigned, so an upgraded node with the default settings no longer answers it
    any record or Standard History.
  - **A node that no record names is refused a record before the subject is looked up,** with the
    same refusal for any subject, so the refusal no longer shows how large the log is.
  - **The replay check is held in memory,** so a restart forgets it: a request can be repeated to
    the same node, before its deadline, across a restart.
  - **Keys no host announced share one replay pool** of 16,384 nonces, each held until its
    deadline, and their replay check is best-effort. While the pool is full of live nonces, their
    signed requests are answered as unsigned requests would be, not refused, and logged as
    `replay` `unchecked`; a replay that is detected is still refused. Keys a host announced each
    have their own room, and their replays are always refused.
  - **The "asked of you" log records each request's digest,** the key and node it was proven to
    come from (or `unbound`), the audience it named, and whether its nonce was checked for
    replay.
- **Fetching a record by id (`ledger-fetch/1`) is proven the same way.** The asker signs the
  fetch, and the node answering checks it before it looks the record up. Unless records go to
  any peer (`peers`), a record goes only to the node the signature proves, never to the id the
  fetch states. Under the default
  `share_history_segments` an unsigned fetch, from a node on 0.1.5 or older, is now refused
  `not_authorized`; under `peers` it is answered as before. With sharing `off`, a fetch is refused
  before the lookup, so the refusal no longer shows whether the record exists.
- **A referee checks the signature on an `adjudicate` request** before it reads anything. A
  request addressed to another node, changed after signing, past its deadline or replayed is
  refused `not_authorized`. A verdict does not depend on who asks, so an unsigned request is
  answered as before.
- **Each kind of signed request has its own domain tag:** `ledger-fetch/1/requester-auth` for a
  fetch and `referee-adjudicate/1/requester-auth` for an `adjudicate` request, so a signature
  made for one is refused on any other, even where its body would be read there. An evidence
  request's tag is unchanged (`evidence-request/1/requester-auth`).
- **The records status (`evidence_records_status`) says whether this node's requests are
  signed** (`requests_signed`). They are not when its host refused its key or lacks
  `plugin_keys.v1`; its requests then go out unsigned.
- **Standard History is no longer answered to every node that asks.** A node's checkpoint list,
  `history_card/1` and `served_summary/1` now go only to a counterparty (a node one of its records
  names as the other side of an exchange). Every other node gets a signed `policy_declined` that
  carries nothing about the asker: it is signed over the request's digest, the reason and the
  time, so the same request gets the same refusal whoever sends it. Before 0.1.6 any node could
  read them. **After upgrading, a node you have never exchanged with can no longer read your
  checkpoints, history card or served summary** until you set the new `share_standard_history`
  switch to `on` (console: Sharing policy › Standard history; or
  `CAPSULES_SHARE_STANDARD_HISTORY=on`). Your records are unaffected: they stay under
  `share_history_segments`. A counterparty is the node a signed request proves it came from
  (see above).
- **`share_history_segments` defaults to `counterparties`.** `prospective`, the old default, is
  deprecated and read as `counterparties`: it always served exactly the same nodes, so nothing
  changes for a node using it. A log line says when it is read. Set `counterparties` instead.

### Added

- **Witness receipts on an answer are checked under the witness keys you configured:** when this
  node asks another for evidence, each witness receipt carried on the answer's checkpoints is
  checked under the key set for that witness's URL in `CAPSULES_CHECKPOINT_WITNESS_KEYS` (URLs
  matched as for your own checkpoints; only keys you set, not keys pinned on first contact). A
  receipt that does not verify under its witness's key makes the answer not evidence, with reason
  `witness_receipt_unverified`. Receipts on a `served_summary/1` answer are not checked yet. With
  keys set, the result also says how many receipts were verified and each receipt's status. A
  key set for one witness is never used for another. With no keys set, nothing changes: receipts
  are checked for form and for the checkpoint they are on, and none counts as witnessed.
- **`advertise_evidence_requests`, off by default.** Whether this node tells other nodes that it
  answers evidence requests. This version advertises nothing, so nothing reads it yet. A node
  with it off still answers a node that asks.
- The plugin's records status (`sharing`) reports both new switches. The Evidence page does not
  show them yet.
- **A TDX quote in the record, on a TDX host:** when the plugin can take a quote (the kernel's
  configfs-tsm report interface, writable by the plugin's user), each record's `tee_attestation`
  slot carries the quote, its SHA-256 and the 64-byte REPORT_DATA. REPORT_DATA is
  `SHA-512("x-mesh-poc-v1/tee-binding/v1" || 0x00 || request digest || answer digest ||
  SHA-256(client nonce) || the record's signing key)`, each input 32 bytes, the nonce hashed
  exactly as recorded. A quote is taken only for a real client nonce (a host-forwarded one or one
  the client sent to capsules' own `/v1`), never for a placeholder; otherwise the slot says "no
  client nonce". The context says the quote is bound to the exchange's nonce, and fresh against
  the client's challenge only when the record shows the client chose that nonce. A host whose report interface is not TDX, or not usable by the
  plugin's user, records the slot empty exactly as before (the latter is said once in the log).
- **What the node checks before it calls a quote bound, all offline:** the quote's SHA-256; that
  it parses as a TDX DCAP v4 quote; that its REPORT_DATA is the one recomputed from the record; that
  the bound key signs the record; the quote's signature by its attestation key; the QE report's
  signature by the PCK key; that the QE report binds the attestation key; and the PCK certificate
  chain to the Intel SGX Root CA, pinned in the code by its SHA-256, at the record's own time. Any
  failure shows as failed. The Evidence page then says "Quote signed by a TDX quoting enclave and
  bound to this exchange (checked here)", and on its own line that platform status, revocation and
  measurements are not checked here. The checks use `p256` 0.13 (ECDSA P-256) and `x509-cert` 0.2.
- **Each node's key is announced by capsules itself:** on a host that lists `plugin_keys.v1`,
  capsules asks the host at start to announce the public key its records are signed with, so a
  peer verifies this node's pushed records without `CAPSULES_PEER_KEYS` naming it. On an older
  host nothing is sent, and the log says the key was not announced. `CAPSULES_PEER_KEYS` stays the fallback; a peer
  the two sources give different keys still verifies nothing until the stale entry is removed.
- **Built against mesh-llm-plugin from mesh-llm main (a39211e),** which has the announce call;
  no published release has it yet.

### Fixed

- **A paid exchange's CLOSED row no longer reads "served by: not recorded" on the payer's
  Evidence page when the payer's own record names the provider.** The row read the node id the
  page is sent only while the other side's record is still to be fetched, so once that record was
  held the row lost it. It now reads the payer's own record (`served_by_node_id`, the same rule the
  plugin uses), and says "not recorded" only when the record names no server.
- **A half from a model served by a Hugging Face reference is no longer refused `model_mismatch`.**
  The digest inside a `local-gguf/sha256-…` model id is the file's own SHA-256 only for a single
  local file. For a Hugging Face GGUF it hashes the repository, revision and file list, so it was
  read as a second, different weights claim. Weights fields are now held to weights fields and
  model ids to model ids. A half whose two weights fields disagree, or that names another model id
  than the one asked for, is still refused, and the Evidence page applies the same rule.

## 0.1.4

### Verified on live hosts, and not

- **Paid exchanges across two nodes: verified.** Four paid exchanges between two nodes built from
  this release, the requesting node serving no model. In each, the response's `x-mesh-served-by`
  and the record's `served_by` named the serving node, and each node sealed its own half.
- **Peers' keys announced by the host:** exercised only against a test build of the host, not on a
  stock mesh-llm release. `CAPSULES_PEER_KEYS` remains the fallback.
- **The referee with texts the client supplies:** stock hosts now name the node that asked
  (`requested_by_node_id`, from mesh-llm a39211e), so nothing is missing on the host side. A
  referee verdict on live hosts waits on 0.1.5; until then it is covered by tests only.

### Added

- **Each paid exchange on the Evidence page shows what it charged and who paid, from the records
  this node sealed:** the wallet's amount and fee on each settlement (and what it credited, on the
  provider side), the payer's own recorded total, and the provider's delivered-token watermark.
  Each value says who stated it; nothing is summed, and no balance is shown.
- **An exchange this node served for pay shows its own book as the provider,** joined by the
  exchange id the host names a provider's payment events with: who paid, what its wallet reported
  receiving, and what it delivered.
- **"Outcome not reported":** an invoice with no settlement the wallet reported and no final amount
  says so. The host emits no event when a payment's outcome is uncertain or it failed, so the page
  doesn't say which, and never says "unpaid". A paid request refused before any payment (for
  example a payment protocol version the other node doesn't speak) leaves no payment records,
  and the page says that this can happen.
- **Settlement records carry the wallet's `credited_msat` and `fee_msat`** when the host's event
  does; a record from an event without them is unchanged.
- **The referee can take a twin pair's texts from the client** (`referee_supply_twin_texts`: the
  request and both answers), since stock mesh-llm passes no exchange text to plugins. They are
  kept only when the referee is on, the request and each answer are the bodies this node's own
  records sealed, and the answers differ.
- **Peers' keys from the host:** when mesh-llm lists `plugin_keys.v1`, the keys it announces for
  this plugin (`GET /api/plugin-keys`) are merged with `CAPSULES_PEER_KEYS`, which stays the
  fallback. A peer the two give different keys verifies nothing until the stale entry is removed.
- **Twin pairs on Peers:** each peer row states its twin pairs as facts ("Twin pairs: N · answers
  differed: M · not adjudicated"), so a peer whose twin answers differed no longer reads clean
  there.

### Changed

- **The plugin reads and hashes its own executable once per process**, at the first record it
  seals, instead of for every record. Each record still carries that measurement, signed with the
  node key and stamped with the record's own time, so records are unchanged. On macOS the kernel's
  code-signing status is still asked for every record, because it can change while the process
  runs. Replacing the plugin's executable on disk while it runs is measured only after a restart,
  which is also when the new binary starts running.
- **A served exchange with no other side reads LOCAL** ("this node only"), not OPEN, and the state
  filter counts it under "This node only".
- **Each choice setting lists its default first,** so mesh-llm 0.78's console, which does not
  receive declared defaults, shows what the plugin actually does for an unset setting.
- **`capsule-emit-evidence-request` is pinned to `=0.0.3`** from crates.io. An answer to an
  evidence request this node sent is now refused when a witness receipt it carries is malformed or
  not bound to its checkpoint, or when an exchange answer's anchor is not the earliest checkpoint
  covering it. This node holds no witness keys for this check, so no receipt in an answer is
  counted as witnessed. 0.0.3 also signs a refusal over a deterministic encoding, so a refusal
  from a 0.1.3 node may not verify on 0.1.4: upgrade both nodes.

### Fixed

- **The console's top navigation stays on the Evidence page.** The page's stylesheet applies only
  to the page's own elements; a global `.hidden` rule had hidden the console's nav.
- **An exchange this node served for pay is not shown as local.** With no payer recorded, its row
  read "Local — no other side"; it now reads "Served for pay — payer not recorded" and counts as
  open. With no payer recorded there is no node to ask for its record, so such a row may stay open
  until the payer sends its own, and its (i) says so.

## 0.1.3

### Changed

- **Nothing reaches a peer on its own by default.** `share_record_at_completion` (push this node's
  record to the counterparty), `share_adjudications` (deliver verdicts) and
  `adjudicate_differing_twins` (ask a referee, which sends the twins' request to a third node) are
  **off by default**; each turns on only on its explicit value. Answering a peer that asks for a
  record (`share_history_segments`) is unchanged.
- **The stop-routing rule asks the host through the plugin protocol** (`PeerBlockRequest`), so the
  host records the block as this plugin's and refuses it unless the operator set
  `allow_peer_blocks = true` for this plugin. The plugin no longer calls the console's
  `/api/peer-blocks` route (as the operator) for this or for choosing a referee; it learns blocks
  from the host's `routing.choice.v1` publications, and seals its own record of a rule block.
- **The twin pair says what happened:** "A client marked these two requests as one pair", not that
  a comparison ran automatically.
- **Renamed to `capsules`** (plugin id, repository `Mesh-LLM/capsules`, release assets
  `capsules-<version>-<target>.tar.gz`, page `/plugins/capsules/evidence`). Settings are
  `CAPSULES_*`; the old `CAPSULE_EMIT_MESH_*` names are read for this release, with a warning. The
  `ADMISSION_POLICY_*` names are no longer read.
- **An upgraded node keeps its data directory in place** under the old name
  (`capsule-emit-mesh`); nothing is moved, so an upgrade never races a writer that is still
  running. A node with both directories refuses to start and says why. Existing logs keep their
  log id. On Linux the plugin refuses a log another process has open.

### Added

- **A list of witnesses.** The console's `witness` setting is a list of `{ endpoint, public_key }`
  rows, the key optional (the environment's `CAPSULES_CHECKPOINT_WITNESS_URLS` stays a comma list,
  with keys in `CAPSULES_CHECKPOINT_WITNESS_KEYS`, a JSON object from URL to key). There is still no default witness, and no
  witness is contacted until one is named. The Evidence page lists each witness by name with what
  it holds.
- **Receipts are checked before they count.** A witness's receipt counts only when it verifies
  offline: the entry recomputed from the checkpoint's signed fields, and the receipt's inclusion
  proof and signature under the witness's key: the one configured, else one fetched once from that
  witness and kept, which the page labels "pinned on first contact, not configured". An
  unchecked or failing receipt is shown, with the reason, and never counted. A witness that does
  not hold the latest checkpoint (or whose receipt for it does not check) says why: timed out,
  refused, an HTTP error, or a different key.
- **The Evidence page can be a primary tab** (mesh-llm 0.78 and later). The page asks for primary
  placement; the operator turns it on in the console's plugin settings ("Primary tab"), and with it
  off the page stays a navigation item. The page draws its own header, so the host's header is
  turned off for it.
- **Under a chat answer and in the Logs request inspector** (mesh-llm 0.78 and later): whether
  this node sealed that exchange, with a link to its record on the Evidence page. The lookup is
  local (`GET /lookup?exchange_id=` or `?client_nonce=`); nothing is shown when it fails or finds
  no record. It works on the node that asked a peer for the answer. For an exchange this node
  served itself, mesh-llm 0.78 does not give the plugin the ids those views hold, so nothing is
  shown there. On mesh-llm 0.77 the plugin installs and runs as before, without these.

### Fixed

- **The log of requests made of this node never stops the plugin starting.** It is written only to
  a directory that exists when the plugin starts (`CAPSULES_RECEIVED_LOG_DIR`, else
  `<data dir>/received-log`) and that this process holds a lock on; never the ledger directory. A
  directory another node holds, or one that cannot be locked, turns that log off for the run, with
  a warning.
- **A witness is dialled at the URL given.** With capsule-emit 0.0.4, one public witness URL
  (`https://witness.agentactioncapsule.org`) was sent to a different host
  (`anchor.agentactioncapsule.org`). capsule-emit 0.0.6, now required, dials every witness exactly as configured and
  follows no redirect, and the plugin's own witness reads do the same.
- **Settings saved in the console reach the plugin.** The plugin reads its `[plugin.settings]`
  from mesh-llm's config file (`MESH_LLM_CONFIG`, or `~/.mesh-llm/config.toml`); the environment
  still wins. The console's `witness` is the checkpoint witness, and the witness client uses it.
- **One plugin process per data directory.** A second process on the same directory refuses to
  start instead of interleaving appends and breaking the chain.

## 0.1.2

### Fixed

- **The plugin serves no models unless asked.** 0.1.1 registered an OpenAI-compatible inference
  provider on every node and, with `CAPSULE_EMIT_MESH_BLOCKED_MODELS` unset or empty, advertised
  `blocked-test-model` through it. That model always answers 403, so a client that takes the first
  model `/v1/models` lists, or "Mesh automatic" routing, could pick it. From 0.1.2 the admission
  endpoint is opt-in: only when `CAPSULE_EMIT_MESH_BLOCKED_MODELS` names models does the plugin
  register the provider and declare `admission_policy.v1`. Unset, empty, `none` or `off`: no
  provider, no admission capability, no models. The plugin seals records and serves the Evidence
  page.

## 0.1.1

### Added

- **Settlement-record legs for paid exchanges** (draft-mih-agent-settlement-records-00). For each
  invoice of a paid exchange, both sides now seal the draft's legs beside the existing payment
  lifecycle trail. The payer seals the terms leg (invoice amount, `ln.payment_hash`) and pushes it
  to the serving peer, then seals `payer_observed` (the amount, plus `routing_fee` when the host
  passes the wallet's fee) and a delivered leg with the response digest. The provider seals
  `payee_observed` (`received`, `receive_fee`), citing the payer's terms leg.
  - **Host requirement:** the provider's leg needs a mesh-llm host whose settlement events carry
    the wallet's credited amount and fee (Mesh-LLM/mesh-llm#2169). On a host without it, the
    provider seals no payee leg; the payer's legs and pushed terms work either way.
  - `routing_fee` and `receive_fee` are new on ledger records. Peers may read them back under the
    default share policy, like the amounts the trail already carried.

### Fixed

- **A node-unique default log id.** Every node's checkpoints used the same log id, so a witness
  accepted the first node and refused the rest as a fork. The default is now
  `capsule-emit-mesh/<key id>`, chosen at first start and kept in `<data dir>/log_id`;
  `CAPSULE_EMIT_MESH_LOG_ID` overrides it. A log that already holds checkpoints keeps its id, and
  starting a new log moves the node to its own id (see INSTALL.md, "The log's id").
- **Finding a served half by its digests.** When the host forwarded no nonce, the serving node's
  half couldn't be found by the requester's nonce, so the exchange showed as open. The responder
  now also indexes each record by its exchange's two effect digests, and the page asks by those
  after a signed "no such record". Both nodes need 0.1.1 for this.
- **The witness indicator.** The Evidence page could say "witness off" on a node whose checkpoints
  carried witness receipts. It now reads how many checkpoints a witness holds and whether a
  witness is configured, and says "off" only when no witness is set.

## 0.1.0

- First release.
