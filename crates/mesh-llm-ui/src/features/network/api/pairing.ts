import { env } from '@/lib/env'
import { ApiError, parseApiErrorBody } from '@/lib/api/errors'

export type PairingSessionStatus =
  | 'connecting'
  | 'awaiting_approval'
  | 'waiting_for_peer'
  | 'joining'
  | 'approved'
  | 'rejected'
  | 'cancelled'
  | 'expired'
  | 'failed'

export type PairingSession = {
  id: string
  direction: 'incoming' | 'outgoing'
  peer_name: string
  peer_id: string
  comparison_code?: string
  status: PairingSessionStatus
  created_at: number
  expires_at: number
  error?: string
}

export type PairingOffer = {
  offer: string
  url: string
  expires_at: number
}

export type NearbyPairingDevice = {
  instance_name: string
  host: string
  published_version?: string
  pairing_offer?: string
  pairing_offer_expires_at?: number
  listing: {
    name?: string
    region?: string
  }
}

async function requestJson<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${env.managementApiUrl}${path}`, init)
  if (!response.ok) {
    const body = await parseApiErrorBody(response)
    throw new ApiError(response.status, body, body)
  }
  return response.json() as Promise<T>
}

export async function fetchPairingSessions(): Promise<PairingSession[]> {
  const payload = await requestJson<{ sessions: PairingSession[] }>('/api/pairing/sessions')
  return payload.sessions
}

export function createPairingOffer(): Promise<PairingOffer> {
  return requestJson('/api/pairing/offers', { method: 'POST' })
}

export function connectWithPairingOffer(offer: string): Promise<PairingSession> {
  return requestJson('/api/pairing/connect', {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ offer })
  })
}

export function decidePairingSession(id: string, decision: 'approve' | 'reject' | 'cancel'): Promise<PairingSession> {
  return requestJson(`/api/pairing/sessions/${encodeURIComponent(id)}/${decision}`, { method: 'POST' })
}

export function discoverNearbyPairingDevices(): Promise<NearbyPairingDevice[]> {
  return requestJson('/api/discover')
}
