import { describe, expect, it } from 'vitest'
import { isOfferedForMode, parseModelPayments, priceLabel } from '@/features/chat/lib/model-prices'

const offer = (paid: boolean, outputMsatPerMillion?: number) => ({
  binding_quote: false,
  paid,
  pricing: paid ? { input_msat_per_million: 1000, output_msat_per_million: outputMsatPerMillion ?? 1500 } : null,
  rate_unit: 'msat_per_million_tokens'
})

const listing = {
  data: [
    {
      id: 'free-model',
      providers: [{ provider_id: 'peer-a', local: false, available: true, payment: offer(false) }]
    },
    {
      id: 'paid-model',
      providers: [
        { provider_id: 'peer-a', local: false, available: true, payment: offer(true, 1500) },
        { provider_id: 'peer-b', local: false, available: true, payment: offer(true, 2500) }
      ]
    },
    {
      id: 'mixed-model',
      providers: [
        { provider_id: 'peer-a', local: false, available: true, payment: offer(true, 900) },
        { provider_id: 'peer-b', local: false, available: true, payment: offer(false) }
      ]
    },
    { id: 'no-payment-data', providers: [{ provider_id: 'peer-a', local: false, available: true }] }
  ]
}

describe('model prices', () => {
  const payments = parseModelPayments(listing)

  it('parses the cheapest paid output rate across providers', () => {
    expect(payments.get('paid-model')).toEqual({
      freeAvailable: false,
      paidAvailable: true,
      outputMsatPerMillion: 1500
    })
    expect(payments.has('no-payment-data')).toBe(false)
    expect(parseModelPayments(null).size).toBe(0)
  })

  it('hides only paid-only models in free mode', () => {
    expect(isOfferedForMode(payments.get('paid-model'), true)).toBe(false)
    expect(isOfferedForMode(payments.get('paid-model'), false)).toBe(true)
    expect(isOfferedForMode(payments.get('mixed-model'), true)).toBe(true)
    expect(isOfferedForMode(undefined, true)).toBe(true)
  })

  it('ignores offline providers when deriving availability and pricing', () => {
    const offline = parseModelPayments({
      data: [
        {
          // A stopped local/free provider must not keep the model free-available.
          id: 'free-gone',
          providers: [{ provider_id: 'self', local: true, available: false, payment: offer(false) }]
        },
        {
          // An offline cheaper paid provider must not set the shown minimum.
          id: 'paid-partial',
          providers: [
            { provider_id: 'peer-a', local: false, available: false, payment: offer(true, 700) },
            { provider_id: 'peer-b', local: false, available: true, payment: offer(true, 1800) }
          ]
        },
        {
          // Every provider offline with no payment data stays unlisted, as before.
          id: 'all-offline',
          providers: [{ provider_id: 'peer-a', local: false, available: false }]
        }
      ]
    })
    expect(offline.get('free-gone')).toBeUndefined()
    expect(offline.get('paid-partial')).toEqual({
      freeAvailable: false,
      paidAvailable: true,
      outputMsatPerMillion: 1800
    })
    expect(offline.has('all-offline')).toBe(false)
  })

  it('keeps locally hosted priced offers free for this node', () => {
    const local = parseModelPayments({
      data: [
        {
          id: 'local',
          providers: [{ provider_id: 'self', local: true, available: true, payment: offer(true) }]
        }
      ]
    }).get('local')
    expect(local).toMatchObject({ freeAvailable: true, paidAvailable: true })
    expect(isOfferedForMode(local, true)).toBe(true)
  })

  it('keys payments by the public discovery identity, profile suffixes included', () => {
    const payments = parseModelPayments({
      data: [
        {
          id: 'org/repo:Q4#pro',
          providers: [{ provider_id: 'peer-a', local: false, available: true, payment: offer(true, 1500) }]
        }
      ]
    })
    expect(isOfferedForMode(payments.get('org/repo:Q4#pro'), true)).toBe(false)
    expect(priceLabel(payments.get('org/repo:Q4#pro'))).toBe('from 1.5k msat/M out')
    expect(payments.has('org/repo:Q4')).toBe(false)
  })

  it('labels paid listings', () => {
    expect(priceLabel(payments.get('paid-model'))).toBe('from 1.5k msat/M out')
    expect(priceLabel(payments.get('mixed-model'))).toBe('free · paid')
    expect(priceLabel(payments.get('free-model'))).toBeUndefined()
  })
})
