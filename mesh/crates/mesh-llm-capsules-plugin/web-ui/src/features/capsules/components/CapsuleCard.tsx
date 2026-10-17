import { useEffect, useState } from 'react'
import { Badge } from '@/components/ui/badge'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { cn } from '@/lib/utils'
import { fetchDisclosurePreimage, fetchSignedStatement } from '@/features/capsules/api/client'
import type { CapsuleRecord, DisclosurePreimage, JsonRecord } from '@/features/capsules/api/types'
import { reconcileAdvertisedVsServed } from '@/features/capsules/lib/advertisement'
import { digestFacts, type JsonValue, recomputeCapsuleId } from '@/features/capsules/lib/canonical'
import { ed25519PublicKeyFromSpkiPem, verifyCoseSign1 } from '@/features/capsules/lib/cose'
import {
  friendlyModelName,
  plainGenParamsLine,
  plainModelLine,
  plainTokenSplit,
  pocBlock,
  servingProvenance
} from '@/features/capsules/lib/serving-provenance'
// buildVerdict and labelCounterparty inlined here after verdict.ts removal (L2 retire)
import type { ServingProvenance, VerdictLine } from '@/features/capsules/api/types'

type VerdictInputs = {
  idMatch: boolean | null
  signatureOk: boolean | null
  hasWitnessCheckpoint: boolean
  counterparty: string
}

function labelCounterparty(record: CapsuleRecord): string {
  const poc = pocBlock(record)
  const crossParty = poc.cross_party as Record<string, unknown> | undefined
  const initiatorRef = crossParty?.initiator_ref
  if (!crossParty || typeof initiatorRef !== 'string' || !initiatorRef) return 'unknown'
  return `initiator:${initiatorRef.slice(0, 12)}`
}

function buildVerdict(sp: ServingProvenance, inputs: VerdictInputs): VerdictLine[] {
  const name = friendlyModelName(sp)
  const gpu = sp.gpu
  let hw = gpu ? ` on ${gpu}` : ''
  if (sp.isSoc && gpu) hw = ` on ${gpu} (Apple silicon)`
  const split = plainTokenSplit(sp)

  let line1: VerdictLine
  if (!sp.model) {
    line1 = { mark: 'warn', text: "No serving-provenance in this record — can't say what ran." }
  } else if (inputs.idMatch === false) {
    line1 = {
      mark: 'warn',
      text: `Attests it ran on ${name}${hw} (self-reported) — but this record's content does NOT recompute to its own capsule_id here. Treat it as altered.`
    }
  } else {
    line1 = {
      mark: 'ok',
      text: `Attests it ran on ${name}${hw} (self-reported)${split ? `, ${split}` : ''} — recompute in-browser that this record is signed+unaltered.`
    }
  }

  let line2: VerdictLine
  if (inputs.signatureOk === true) {
    line2 = {
      mark: inputs.hasWitnessCheckpoint ? 'ok' : 'warn',
      text: inputs.hasWitnessCheckpoint
        ? 'Signed by the provider (verified here) and anchored to a public witness, so it can’t be quietly changed.'
        : 'Provider-signed — the signature verifies here against the node’s public key — but the witness receipt isn’t in this bundle, so anchoring isn’t shown in this view.'
    }
  } else if (inputs.signatureOk === false) {
    line2 = {
      mark: 'warn',
      text: 'Signature does NOT verify against the node’s public key here — treat this record as unauthenticated.'
    }
  } else {
    line2 = {
      mark: 'warn',
      text: 'No signed statement / node public key found in this bundle, so the provider signature can’t be checked here.'
    }
  }

  const line3: VerdictLine =
    inputs.counterparty && inputs.counterparty !== 'unknown'
      ? { mark: 'ok', text: `Who asked is attested (${inputs.counterparty}).` }
      : {
          mark: 'warn',
          text: 'Open gap: who asked is self-attested (not independently verified) — and this node’s track record isn’t carried in a single record.'
        }

  return [line1, line2, line3]
}

type CapsuleCardProps = {
  record: CapsuleRecord
  nodePubKeyPem: string | null
}

type DigestCheck = { sealedDigest: string | null; computedDigest: string | null; matches: boolean | null }

export function CapsuleCard({ record, nodePubKeyPem }: CapsuleCardProps) {
  const capsuleId = record.capsule_id ?? '(none)'
  const sp = servingProvenance(record)
  const friendlyModel = friendlyModelName(sp)
  const modelLine = plainModelLine(sp)
  const genParamsLine = plainGenParamsLine(sp)
  const counterparty = labelCounterparty(record)

  const [recomputedId, setRecomputedId] = useState<string | null>(null)
  const [idMatch, setIdMatch] = useState<boolean | null>(null)
  const [signatureOk, setSignatureOk] = useState<boolean | null>(null)
  const [responseCheck, setResponseCheck] = useState<DigestCheck | null>(null)
  const [disclosure, setDisclosure] = useState<DisclosurePreimage | null>(null)
  const [requestBodyCheck, setRequestBodyCheck] = useState<DigestCheck | null>(null)
  const [responseBodyCheck, setResponseBodyCheck] = useState<DigestCheck | null>(null)

  useEffect(() => {
    let cancelled = false
    async function verify() {
      try {
        const recomputed = await recomputeCapsuleId(record)
        if (cancelled) return
        setRecomputedId(recomputed)
        setIdMatch(record.capsule_id ? recomputed === record.capsule_id : null)
      } catch {
        if (!cancelled) setIdMatch(null)
      }

      if (nodePubKeyPem && record.capsule_id) {
        try {
          const statementBytes = await fetchSignedStatement(record.capsule_id)
          if (cancelled) return
          if (statementBytes) {
            const publicKey = ed25519PublicKeyFromSpkiPem(nodePubKeyPem)
            const result = verifyCoseSign1(statementBytes, publicKey)
            setSignatureOk(result.verified)
          } else {
            setSignatureOk(null)
          }
        } catch {
          if (!cancelled) setSignatureOk(null)
        }
      }

      const disclosed = record.capsule_id ? await fetchDisclosurePreimage(record.capsule_id) : null
      if (cancelled) return
      setDisclosure(disclosed)

      const effect = (record.effect as JsonRecord | undefined) ?? {}
      const sealedRequestDigest = typeof effect.request_digest === 'string' ? effect.request_digest : null
      const sealedResponseDigest = typeof effect.response_digest === 'string' ? effect.response_digest : null

      // Request body: only ever checkable when the sidecar disclosed it --
      // there is no served-facts fallback for the request side.
      if (disclosed?.request_body) {
        try {
          const computed = await digestFacts(disclosed.request_body as JsonValue)
          if (!cancelled) {
            setRequestBodyCheck({
              sealedDigest: sealedRequestDigest,
              computedDigest: computed,
              matches: sealedRequestDigest != null && computed === sealedRequestDigest
            })
          }
        } catch {
          if (!cancelled)
            setRequestBodyCheck({ sealedDigest: sealedRequestDigest, computedDigest: null, matches: null })
        }
      }

      // Response: a disclosed response BODY is the strongest, byte-exact
      // check (matches capsule_mesh_viewer.build_conversation) -- prefer it.
      // Only fall back to the served-facts approximation (model + token
      // usage) when no disclosed body is present, since on some sealing
      // paths response_digest commits to the full body, not just the facts,
      // and comparing facts against a full-body digest would be a false
      // mismatch, not an honest check.
      if (disclosed?.response_body) {
        try {
          const computed = await digestFacts(disclosed.response_body as JsonValue)
          if (!cancelled) {
            setResponseBodyCheck({
              sealedDigest: sealedResponseDigest,
              computedDigest: computed,
              matches: sealedResponseDigest != null && computed === sealedResponseDigest
            })
          }
        } catch {
          if (!cancelled)
            setResponseBodyCheck({ sealedDigest: sealedResponseDigest, computedDigest: null, matches: null })
        }
      } else if (sp.model != null && sp.promptTokens != null && sp.completionTokens != null && sp.totalTokens != null) {
        try {
          const facts = {
            model: sp.model,
            usage: {
              prompt_tokens: sp.promptTokens,
              completion_tokens: sp.completionTokens,
              total_tokens: sp.totalTokens
            }
          }
          const computed = await digestFacts(facts)
          if (!cancelled) {
            setResponseCheck({
              sealedDigest: sealedResponseDigest,
              computedDigest: computed,
              matches: sealedResponseDigest != null && computed === sealedResponseDigest
            })
          }
        } catch {
          if (!cancelled) setResponseCheck({ sealedDigest: sealedResponseDigest, computedDigest: null, matches: null })
        }
      } else if (sealedResponseDigest) {
        setResponseCheck({ sealedDigest: sealedResponseDigest, computedDigest: null, matches: null })
      }
    }
    void verify()
    return () => {
      cancelled = true
    }
    // Depend on the whole `record`, not just its id: a malicious relay can
    // re-send tampered content under the same capsule_id on a re-poll, and the
    // verification (id recompute + signature check) must re-run when any field
    // changes, not only when the id does. (ported from mesh-pane-a cdc2dd778)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [record, nodePubKeyPem])

  const verdict = buildVerdict(sp, {
    idMatch,
    signatureOk,
    hasWitnessCheckpoint: false,
    counterparty
  })

  const poc = pocBlock(record)
  const advertisement = poc.advertisement as JsonRecord | undefined
  const rawServingProvenance = poc.serving_provenance as JsonRecord | undefined
  const reconciliation = reconcileAdvertisedVsServed(advertisement, rawServingProvenance)

  const sealedLabel =
    idMatch === true
      ? 'sealed — integrity confirmed here'
      : idMatch === false
        ? 'sealed — MISMATCH'
        : 'sealed — checking…'
  const sealedTone = idMatch === true ? 'ok' : idMatch === false ? 'fail' : 'muted'

  return (
    <Card className="mb-4">
      <CardHeader className="flex-row items-baseline justify-between gap-3">
        <CardTitle className="text-base">{friendlyModel}</CardTitle>
        <Badge
          className={cn(
            sealedTone === 'ok' && 'border-emerald-600/40 bg-emerald-600/10 text-emerald-500',
            sealedTone === 'fail' && 'border-red-600/40 bg-red-600/10 text-red-500'
          )}
        >
          {sealedLabel}
        </Badge>
      </CardHeader>
      <CardContent className="space-y-3">
        <ul className="space-y-1.5">
          {verdict.map((line, idx) => (
            <li key={idx} className={cn('flex gap-2 text-sm leading-snug', line.mark === 'warn' && 'text-amber-500')}>
              <span aria-hidden="true">{line.mark === 'ok' ? '✓' : '⚠'}</span>
              <span>{line.text}</span>
            </li>
          ))}
        </ul>

        <div className="rounded-md border border-border/60 bg-muted/30 p-2 font-mono text-xs text-muted-foreground">
          <div>recorded (self-reported): {modelLine}</div>
          {genParamsLine ? <div className="mt-1">{genParamsLine}</div> : null}
        </div>

        <AdvertisedVsServed reconciliation={reconciliation} />

        <DisclosurePanel disclosure={disclosure} requestCheck={requestBodyCheck} responseCheck={responseBodyCheck} />

        <details className="rounded-md border border-border/60 text-xs">
          <summary className="cursor-pointer select-none px-3 py-2 font-medium text-muted-foreground">
            Show the security checks
          </summary>
          <div className="space-y-1 px-3 pb-3 font-mono text-muted-foreground">
            <div>capsule_id: {capsuleId}</div>
            <div>recomputed in-browser: {recomputedId ?? '(not recomputable)'}</div>
            <div>id matches: {idMatch === true ? 'yes' : idMatch === false ? 'NO' : 'n/a'}</div>
            <div>
              signature (COSE_Sign1 vs node pubkey):{' '}
              {signatureOk === true ? 'verifies' : signatureOk === false ? 'DOES NOT VERIFY' : 'not checkable here'}
            </div>
            <div>raw model ref: {sp.modelCanonicalRef ?? sp.model ?? '(no raw ref)'}</div>
            {sp.modelIdentityHash ? <div>model_identity_hash: {sp.modelIdentityHash}</div> : null}
            {sp.servedByNodeId ? <div>served_by_node_id: {sp.servedByNodeId}</div> : null}
            {requestBodyCheck ? (
              <div>
                disclosed request body digest vs sealed request_digest:{' '}
                {requestBodyCheck.matches === true
                  ? 'matches'
                  : requestBodyCheck.matches === false
                    ? 'MISMATCH'
                    : 'sealed, not checkable'}
              </div>
            ) : null}
            {responseBodyCheck ? (
              <div>
                disclosed response body digest vs sealed response_digest:{' '}
                {responseBodyCheck.matches === true
                  ? 'matches'
                  : responseBodyCheck.matches === false
                    ? 'MISMATCH'
                    : 'sealed, not checkable'}
              </div>
            ) : responseCheck ? (
              <div>
                served-facts digest vs sealed response_digest:{' '}
                {responseCheck.matches === true
                  ? 'matches'
                  : responseCheck.matches === false
                    ? 'MISMATCH'
                    : 'sealed, not checkable'}
              </div>
            ) : null}
          </div>
        </details>
      </CardContent>
    </Card>
  )
}

/**
 * The actual request/response TEXT, when a capsule-emit-mesh sidecar
 * disclosed it (capsule-emit-mesh PR #79) -- shown alongside a real,
 * byte-exact recompute-and-match against the sealed request_digest /
 * response_digest, never just displayed on trust. Most capsules (anything
 * sealed before the disclosure feature, or sealed with --no-disclose) have no
 * preimage and stay in the honest "sealed — digest only" default.
 */
function DisclosurePanel({
  disclosure,
  requestCheck,
  responseCheck
}: {
  disclosure: DisclosurePreimage | null
  requestCheck: DigestCheck | null
  responseCheck: DigestCheck | null
}) {
  if (!disclosure) {
    return (
      <div className="rounded-md border border-border/60 p-3">
        <div className="flex items-center justify-between gap-3">
          <span className="text-sm font-medium">Disclosure</span>
          <Badge className="border-border/60 bg-muted/40 text-muted-foreground">sealed — digest only</Badge>
        </div>
        <p className="mt-1.5 text-xs text-muted-foreground">
          Request/response text: sealed — digest only, not disclosed in this bundle. The signed capsule commits to the
          exchange by digest only; no local preimage file was found for this capsule_id.
        </p>
      </div>
    )
  }

  return (
    <div className="rounded-md border border-border/60 p-3">
      <div className="flex items-center justify-between gap-3">
        <span className="text-sm font-medium">Disclosure</span>
        <Badge className="border-emerald-600/40 bg-emerald-600/10 text-emerald-500">disclosed</Badge>
      </div>
      <div className="mt-2 space-y-3">
        <DisclosureField
          label="Request"
          text={disclosure.request_text ?? null}
          hasBody={disclosure.request_body != null}
          check={requestCheck}
          digestLabel="request_digest"
        />
        <DisclosureField
          label="Response"
          text={disclosure.response_text ?? null}
          hasBody={disclosure.response_body != null}
          check={responseCheck}
          digestLabel="response_digest"
          note={disclosure.tool_calls_note ?? null}
        />
      </div>
    </div>
  )
}

function DisclosureField({
  label,
  text,
  hasBody,
  check,
  digestLabel,
  note
}: {
  label: string
  text: string | null
  hasBody: boolean
  check: DigestCheck | null
  digestLabel: string
  note?: string | null
}) {
  return (
    <div>
      <div className="text-xs font-medium text-muted-foreground">{label}</div>
      {text ? (
        <p className="mt-1 whitespace-pre-wrap rounded bg-muted/30 p-2 text-sm leading-snug">{text}</p>
      ) : (
        <p className="mt-1 text-xs text-muted-foreground">(no text extracted)</p>
      )}
      {note ? <p className="mt-1 text-xs text-muted-foreground">{note}</p> : null}
      {hasBody ? (
        <p
          className={cn(
            'mt-1 text-xs',
            check?.matches === true && 'text-emerald-500',
            check?.matches === false && 'text-red-500',
            check?.matches == null && 'text-muted-foreground'
          )}
        >
          {check?.matches === true
            ? `✓ matches the sealed ${digestLabel}`
            : check?.matches === false
              ? `✗ does NOT match the sealed ${digestLabel} — this text is not what was sealed`
              : `checking against sealed ${digestLabel}…`}
        </p>
      ) : null}
    </div>
  )
}

function AdvertisedVsServed({ reconciliation }: { reconciliation: ReturnType<typeof reconcileAdvertisedVsServed> }) {
  const { overall, mismatches } = reconciliation
  if (overall === 'match') {
    return (
      <p className="text-xs text-muted-foreground">
        Advertised vs. served: <span className="text-emerald-500">match</span> — no broken promise in what was both
        claimed and served (both self-attested by this node).
      </p>
    )
  }
  if (overall === 'mismatch') {
    return (
      <p className="text-xs text-amber-500">
        Advertised vs. served: <span className="font-semibold">MISMATCH</span> ({mismatches.join(', ')}) — this node
        advertised one thing and served another.
      </p>
    )
  }
  const why =
    overall === 'advertisement_absent'
      ? 'no advertisement was co-carried to reconcile against'
      : 'no served facts to reconcile'
  return (
    <p className="text-xs text-muted-foreground">
      Advertised vs. served: not checked — {why}, so this is recorded but there is no kept-or-broken promise to check
      (not a pass).
    </p>
  )
}
