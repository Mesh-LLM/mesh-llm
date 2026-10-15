# Known differences: capsule-emit `evidence_request.py` vs draft-mih-agent-evidence-request-00

The Python responder in
[capsule-emit](https://github.com/action-state-group/capsule-emit)
(`capsule_emit/evidence_request.py`) was written against an earlier shape of the
interaction. It differed from the -00 text in the six ways below. The vectors in
this directory are derived from the -00 text, so on each open point that module
fails them.

Checked against capsule-emit `origin/main` at commit
`16f9659375e562781d3c9e64260d32c189dd2f89`. Line numbers refer to that commit.

Items 1 and 2 are resolved in capsule-emit commit
`0f367033910a287b48866125a016581d4c7c9599` (capsule-emit#220): the responder
emits `no_such_subject` and `derivation_unsupported`, and no longer describes a
signed refusal as an absence. Items 3 to 6 are still open.

| # | capsule-emit (file:line) | -00 | Vectors that catch it |
|---|---|---|---|
| 1 | **Resolved** (capsule-emit#220). Refusal reason `no_such_record` (`capsule_emit/evidence_request.py:109-112`; also returned at `:493`, `:554`, `:559`, `:564`). The module's reason set is `{request_malformed, coverage_unsatisfiable, no_such_record}`. | The reason must be one token from the registry (§4.2 item 2, §13). The -00 token for "the subject does not resolve" is `no_such_subject`. `no_such_record` is not registered. | `refusal.json` `neg-unregistered-reason-no_such_record`; `resolution.json` `res-record-not-held`, `res-correlation-prefix`, `res-exchange-not-cited` |
| 2 | **Resolved** (capsule-emit#220). A `no_such_record` refusal is described as "the recorded absence case" and "the wire's recorded_absence" (`:31-32`, `:109`, `:311-314`). | A signed refusal is a refusal. Absence is the requester's own record that nothing arrived, never something a responder sends, and a refusal is never recorded as absence (§4, §4.4, §4.5). | `outcomes.json` `neg-refusal-as-absence`, `neg-unregistered-reason-as-absence`; `refusal.json` `neg-reason-recorded_absence` |
| 3 | Subject kinds `{record, range, chain_segment, correlation}`, carried as `subject.kind` plus per-kind members (`:114`, `:220-238`). | Six forms: `full_history`, `checkpoints`, `record`, `range`, `correlation`, `exchange` (§3.1). `chain_segment` is not one of them, and `full_history`, `checkpoints` and `exchange` are missing. | `request.json` `pos-subject-full_history`, `pos-subject-checkpoints`, `pos-subject-exchange`, `neg-subject-unknown-form`, `neg-subject-kind-member` |
| 4 | Coverage is optional (`data.get("coverage") or {}`, `:240`), and `expected_pin` and `min_freshness` are checked independently, so both or neither pass (`:242-253`). | `coverage` is REQUIRED and carries exactly one of the two members. Both or neither MUST be refused `coverage_unsatisfiable` (§3, §3.2). | `request.json` `neg-coverage-both`, `neg-coverage-neither`, `neg-coverage-missing` |
| 5 | A record id matches by prefix of 8 or more characters (`:383-388`). | A responder MUST NOT attempt fuzzy or best-effort matching; a subject that does not resolve exactly is refused `no_such_subject` (§3.1). | `request.json` `neg-record-digest-prefix`; `resolution.json` `res-record-prefix`, `res-correlation-prefix`, `res-correlation-case-folded` |
| 6 | `expected_pin` is a map `{root, mmr_size}` and `min_freshness` is `{max_age_seconds}` (`:244-253`). | `expected_pin` is a digest; `min_freshness` is a size or a time (§3.2). | `request.json` `neg-expected-pin-map`, `neg-min-freshness-max-age`; `resolution.json` `res-min-freshness-*` |

Notes:

- Item 3 rests partly on this directory's reading of the subject wire shape
  (README ambiguity **A1**). The missing forms and `chain_segment` are
  differences under any reading.
- Item 6 rests partly on ambiguity **A3** (the value shape of `min_freshness`).
  Under any reading, a maximum age is not the same constraint as a minimum
  size or an earliest issuance time.
- The module's refusal signature (`:322-331`: Ed25519 over sorted, compact JSON
  of `issued_at`, `reason` and `request_digest`) matches the signature profile
  these vectors use (README ambiguity **A7**). The imported interop vector
  `refusal-imported.json` was produced by Python code using this same signing body.
- The module's `page` request member is a deployment extension. -00 requires
  responders to ignore unknown fields (§3), so it is not a difference.
