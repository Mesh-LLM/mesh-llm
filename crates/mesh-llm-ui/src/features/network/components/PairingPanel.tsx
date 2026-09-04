import { useEffect, useMemo, useState } from 'react'
import QRCode from 'qrcode'
import { Check, Copy, Link2, Loader2, QrCode, Radar, RefreshCw, Send, ShieldCheck, X } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { StatusBadge, type StatusBadgeTone } from '@/components/ui/StatusBadge'
import {
  useConnectPairingOffer,
  useCreatePairingOffer,
  useNearbyPairingDevices,
  usePairingDecision,
  usePairingSessions
} from '@/features/network/api/use-pairing'
import type { PairingSession } from '@/features/network/api/pairing'
import { copyStateLabel } from '@/lib/copyStateLabel'
import { useClipboardCopy } from '@/lib/useClipboardCopy'

type PairingMode = 'nearby' | 'invite' | 'connect'

function pairingStatus(session: PairingSession): { label: string; tone: StatusBadgeTone } {
  switch (session.status) {
    case 'awaiting_approval':
      return { label: 'Needs approval', tone: 'warn' }
    case 'waiting_for_peer':
      return { label: 'Waiting for other device', tone: 'accent' }
    case 'joining':
      return { label: 'Joining mesh', tone: 'accent' }
    case 'approved':
      return { label: 'Connected', tone: 'good' }
    case 'rejected':
      return { label: 'Rejected', tone: 'bad' }
    case 'cancelled':
      return { label: 'Cancelled', tone: 'muted' }
    case 'expired':
      return { label: 'Expired', tone: 'muted' }
    case 'failed':
      return { label: 'Could not connect', tone: 'bad' }
    default:
      return { label: 'Connecting', tone: 'accent' }
  }
}

function comparisonCode(code?: string) {
  if (!code) return undefined
  return `${code.slice(0, 3)} ${code.slice(3)}`
}

function errorMessage(error: unknown) {
  if (error instanceof Error) return error.message
  return 'The local Mesh service could not complete that action.'
}

function useUnixTime(enabled: boolean) {
  const [now, setNow] = useState(0)

  useEffect(() => {
    if (!enabled) return
    const update = () => setNow(Math.floor(Date.now() / 1000))
    const initial = window.setTimeout(update, 0)
    const interval = window.setInterval(update, 1_000)
    return () => {
      window.clearTimeout(initial)
      window.clearInterval(interval)
    }
  }, [enabled])

  return now
}

function expiryLabel(expiresAt: number, now: number) {
  if (!now || expiresAt <= now) return 'Expired'
  const seconds = expiresAt - now
  if (seconds < 60) return `Expires in ${seconds} second${seconds === 1 ? '' : 's'}`
  const minutes = Math.ceil(seconds / 60)
  return `Expires in ${minutes} minute${minutes === 1 ? '' : 's'}`
}

function PairingSessionRow({ session }: { session: PairingSession }) {
  const decision = usePairingDecision()
  const status = pairingStatus(session)
  const code = comparisonCode(session.comparison_code)
  const canDecide = session.status === 'awaiting_approval'
  const canCancel = ['connecting', 'waiting_for_peer', 'joining'].includes(session.status)

  return (
    <article className="grid gap-3 border-b border-border-soft px-4 py-3 last:border-b-0 md:grid-cols-[minmax(0,1fr)_auto] md:items-center">
      <div className="min-w-0">
        <div className="flex flex-wrap items-center gap-2">
          <span className="type-body font-semibold text-foreground">{session.peer_name}</span>
          <span role="status" aria-live="polite">
            <StatusBadge dot tone={status.tone}>
              {status.label}
            </StatusBadge>
          </span>
          <span className="type-caption text-fg-faint">
            {session.direction === 'incoming' ? 'wants to join' : 'you are joining'}
          </span>
        </div>
        {code ? (
          <div className="mt-2 flex flex-wrap items-baseline gap-2">
            <span className="font-mono text-[length:var(--density-type-title)] font-semibold tracking-[0.14em] text-foreground">
              {code}
            </span>
            <span className="type-caption max-w-[58ch] text-fg-dim">
              Approve only if this code matches on the other device.
            </span>
          </div>
        ) : null}
        {session.error ? (
          <p className="type-caption mt-1.5 text-bad" role="alert">
            {session.error}
          </p>
        ) : null}
      </div>
      {canDecide ? (
        <div className="flex items-center gap-2">
          <Button
            variant="outline"
            size="sm"
            disabled={decision.isPending}
            onClick={() => decision.mutate({ id: session.id, decision: 'reject' })}
          >
            <X className="size-3.5" aria-hidden="true" />
            Reject
          </Button>
          <Button
            size="sm"
            disabled={decision.isPending}
            onClick={() => decision.mutate({ id: session.id, decision: 'approve' })}
          >
            <ShieldCheck className="size-3.5" aria-hidden="true" />
            Codes match
          </Button>
        </div>
      ) : canCancel ? (
        <Button
          variant="outline"
          size="sm"
          disabled={decision.isPending}
          onClick={() => decision.mutate({ id: session.id, decision: 'cancel' })}
        >
          <X className="size-3.5" aria-hidden="true" />
          Cancel
        </Button>
      ) : null}
    </article>
  )
}

function InviteView({ now }: { now: number }) {
  const createOffer = useCreatePairingOffer()
  const { copyState, copyText } = useClipboardCopy()
  const [qr, setQr] = useState<{ inviteUrl: string; dataUrl?: string; failed?: boolean }>()

  useEffect(() => {
    const inviteUrl = createOffer.data?.url
    if (!inviteUrl) return
    let active = true
    void QRCode.toDataURL(inviteUrl, { width: 180, margin: 1 })
      .then((value) => {
        if (active) setQr({ inviteUrl, dataUrl: value })
      })
      .catch(() => {
        if (active) setQr({ inviteUrl, failed: true })
      })
    return () => {
      active = false
    }
  }, [createOffer.data?.url])

  if (!createOffer.data) {
    return (
      <div className="flex min-h-36 flex-col items-start justify-center gap-3 px-4 py-5">
        <div className="flex items-center gap-2 text-foreground">
          <QrCode className="size-4 text-accent" aria-hidden="true" />
          <span className="type-body font-semibold">Create a private pairing invite</span>
        </div>
        <p className="type-body max-w-[68ch] text-fg-dim">
          Send it through Messages, Signal, or email. It expires in ten minutes, works once, and never contains your
          mesh admission token.
        </p>
        <Button disabled={createOffer.isPending} onClick={() => createOffer.mutate()}>
          {createOffer.isPending ? (
            <Loader2 className="size-3.5 animate-spin" aria-hidden="true" />
          ) : (
            <Link2 className="size-3.5" aria-hidden="true" />
          )}
          Create invite
        </Button>
        {createOffer.isError ? <p className="type-caption text-bad">{errorMessage(createOffer.error)}</p> : null}
      </div>
    )
  }

  const shareInvite = async () => {
    if (createOffer.data.expires_at <= now) return
    if (typeof navigator.share === 'function') {
      try {
        await navigator.share({ title: 'Pair with my Mesh device', url: createOffer.data.url })
        return
      } catch (error) {
        if (error instanceof DOMException && error.name === 'AbortError') return
      }
    }
    await copyText(createOffer.data.url)
  }
  const currentQr = qr?.inviteUrl === createOffer.data.url ? qr : undefined
  const expired = now > 0 && createOffer.data.expires_at <= now

  return (
    <div className="grid gap-4 p-4 md:grid-cols-[minmax(0,1fr)_180px]">
      <div className="min-w-0">
        <div className="type-label mb-1.5 text-fg-faint">
          Single-use invite · {expiryLabel(createOffer.data.expires_at, now).toLowerCase()}
        </div>
        <div className="rounded-[var(--radius)] border border-border bg-panel-strong p-3">
          <p className="break-all font-mono text-[length:var(--density-type-caption)] text-fg-dim">
            {createOffer.data.url}
          </p>
        </div>
        <div className="mt-3 flex flex-wrap gap-2">
          <Button size="sm" disabled={expired} onClick={() => void shareInvite()}>
            <Send className="size-3.5" aria-hidden="true" />
            Share invite
          </Button>
          <Button variant="outline" size="sm" disabled={expired} onClick={() => void copyText(createOffer.data.url)}>
            {copyState === 'copied' ? (
              <Check className="size-3.5" aria-hidden="true" />
            ) : (
              <Copy className="size-3.5" aria-hidden="true" />
            )}
            {copyStateLabel(copyState)}
          </Button>
          <Button variant="ghost" size="sm" onClick={() => createOffer.reset()}>
            {expired ? 'Create a new invite' : 'Create another invite'}
          </Button>
        </div>
        <p className="type-caption mt-3 max-w-[65ch] text-fg-faint">
          Both people will still compare a six-digit code and approve before Mesh transfers admission material over
          encrypted QUIC. Creating another invite does not revoke this one; it remains valid until used or expired.
        </p>
      </div>
      <div className="flex min-h-45 items-center justify-center rounded-[var(--radius)] border border-border bg-white p-2">
        {currentQr?.dataUrl ? (
          <img className="size-40" src={currentQr.dataUrl} alt="Pairing invite QR code" />
        ) : currentQr?.failed ? (
          <p className="type-caption max-w-32 text-center text-fg-dim">
            QR unavailable. Share the invite link instead.
          </p>
        ) : (
          <Loader2 className="size-5 animate-spin text-fg-faint" aria-label="Generating QR code" />
        )}
      </div>
    </div>
  )
}

function NearbyView({ enabled, now }: { enabled: boolean; now: number }) {
  const nearby = useNearbyPairingDevices(enabled)
  const connect = useConnectPairingOffer()
  const devices = (nearby.data ?? []).filter(
    (device) =>
      device.pairing_offer && device.pairing_offer_expires_at && now > 0 && device.pairing_offer_expires_at > now
  )

  return (
    <div className="min-h-36">
      <div className="flex items-center justify-between gap-3 border-b border-border-soft px-4 py-2.5">
        <p className="type-caption text-fg-dim">Devices advertising secure pairing on this network</p>
        <Button variant="ghost" size="sm" disabled={nearby.isFetching} onClick={() => void nearby.refetch()}>
          <RefreshCw className={`size-3.5 ${nearby.isFetching ? 'animate-spin' : ''}`} aria-hidden="true" />
          Scan again
        </Button>
      </div>
      {nearby.isLoading ? (
        <div className="flex min-h-28 items-center justify-center gap-2 text-fg-dim" role="status">
          <Radar className="size-4 animate-pulse" aria-hidden="true" />
          <span className="type-body">Scanning nearby devices…</span>
        </div>
      ) : devices.length === 0 ? (
        <div className="flex min-h-28 flex-col items-center justify-center gap-1 px-4 text-center">
          <span className="type-body font-medium text-foreground">No pairing-ready devices found</span>
          <span className="type-caption text-fg-faint">
            Open Mesh on the other computer, or send it an invite instead.
          </span>
        </div>
      ) : (
        <div>
          {devices.map((device) => (
            <div
              key={device.instance_name}
              className="flex flex-wrap items-center justify-between gap-3 border-b border-border-soft px-4 py-3 last:border-b-0"
            >
              <div className="min-w-0">
                <div className="type-body truncate font-semibold text-foreground">
                  {device.listing.name || device.host}
                </div>
                <div className="type-caption text-fg-faint">
                  Nearby{device.published_version ? ` · Mesh ${device.published_version}` : ''}
                </div>
              </div>
              <Button
                size="sm"
                disabled={connect.isPending}
                onClick={() => device.pairing_offer && connect.mutate(device.pairing_offer)}
              >
                Connect
              </Button>
            </div>
          ))}
        </div>
      )}
      {nearby.isError ? (
        <p className="type-caption px-4 py-3 text-bad" role="alert">
          {errorMessage(nearby.error)}
        </p>
      ) : null}
      {connect.isError ? (
        <p className="type-caption px-4 py-3 text-bad" role="alert">
          {errorMessage(connect.error)}
        </p>
      ) : null}
    </div>
  )
}

function ConnectView() {
  const connect = useConnectPairingOffer()
  const [offer, setOffer] = useState(() => {
    if (typeof window === 'undefined') return ''
    return new URLSearchParams(window.location.search).get('pair') ?? ''
  })

  return (
    <form
      className="p-4"
      onSubmit={(event) => {
        event.preventDefault()
        if (offer.trim()) connect.mutate(offer.trim())
      }}
    >
      <label htmlFor="pairing-offer" className="type-label text-fg-faint">
        Pairing invite
      </label>
      <textarea
        id="pairing-offer"
        value={offer}
        onChange={(event) => setOffer(event.target.value)}
        placeholder="Paste a mesh-llm://pair/… invite"
        className="mt-1.5 min-h-24 w-full resize-y rounded-[var(--radius)] border border-border bg-panel-strong px-3 py-2.5 font-mono text-[length:var(--density-type-control)] text-foreground outline-none placeholder:text-fg-faint focus:border-accent focus:ring-2 focus:ring-accent/20"
      />
      <div className="mt-3 flex flex-wrap items-center gap-3">
        <Button type="submit" disabled={!offer.trim() || connect.isPending}>
          {connect.isPending ? (
            <Loader2 className="size-3.5 animate-spin" aria-hidden="true" />
          ) : (
            <ShieldCheck className="size-3.5" aria-hidden="true" />
          )}
          Start secure pairing
        </Button>
        <span className="type-caption text-fg-faint">The invite locates the device; it does not grant access.</span>
      </div>
      {connect.isError ? (
        <p className="type-caption mt-2 text-bad" role="alert">
          {errorMessage(connect.error)}
        </p>
      ) : null}
    </form>
  )
}

export function PairingPanel({ enabled }: { enabled: boolean }) {
  const initialMode = useMemo<PairingMode>(() => {
    if (typeof window === 'undefined') return 'nearby'
    return new URLSearchParams(window.location.search).has('pair') ? 'connect' : 'nearby'
  }, [])
  const [mode, setMode] = useState<PairingMode>(initialMode)
  const now = useUnixTime(enabled)
  const sessions = usePairingSessions(enabled)
  const visibleSessions = sessions.data ?? []

  if (!enabled) return null

  return (
    <section
      id="pairing"
      className="panel-shell overflow-hidden rounded-[var(--radius-lg)] border border-border bg-panel"
    >
      <header className="flex flex-wrap items-center justify-between gap-3 border-b border-border-soft px-4 py-3">
        <div>
          <h2 className="type-panel-title">Add a device</h2>
          <p className="type-caption mt-0.5 text-fg-faint">Find it nearby or invite someone anywhere.</p>
        </div>
        <div
          className="inline-flex rounded-[var(--radius)] border border-border bg-panel-strong p-0.5"
          role="group"
          aria-label="Pairing method"
        >
          {(['nearby', 'invite', 'connect'] as const).map((value) => (
            <button
              key={value}
              type="button"
              aria-pressed={mode === value}
              onClick={() => setMode(value)}
              className={`ui-control h-7 rounded-[calc(var(--radius)-2px)] border-0 px-2.5 text-[length:var(--density-type-caption)] font-medium ${mode === value ? 'bg-surface text-foreground shadow-sm' : 'bg-transparent text-fg-dim'}`}
            >
              {value === 'nearby' ? 'Nearby' : value === 'invite' ? 'Invite someone' : 'Use invite'}
            </button>
          ))}
        </div>
      </header>
      {visibleSessions.length > 0 ? (
        <div className="border-b border-border bg-surface/40" aria-live="polite" aria-relevant="additions text">
          {visibleSessions.map((session) => (
            <PairingSessionRow key={session.id} session={session} />
          ))}
        </div>
      ) : null}
      {mode === 'nearby' ? (
        <NearbyView enabled={enabled} now={now} />
      ) : mode === 'invite' ? (
        <InviteView now={now} />
      ) : (
        <ConnectView />
      )}
    </section>
  )
}
