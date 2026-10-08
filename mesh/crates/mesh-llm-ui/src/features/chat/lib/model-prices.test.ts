import { describe, expect, it } from 'vitest'
import { isOfferedForMode, parseModelPayments, priceLabel } from '@/features/chat/lib/model-prices'

const listing = {
  data: [
    {
      id: 'free-model',
      payment: { free_available: true, paid_available: false, offers: [{ paid: false, pricing: null }] }
    },
    {
      id: 'paid-model',
      payment: {
        free_available: false,
        paid_available: true,
        offers: [
          { paid: true, pricing: { output_msat_per_million: 1500 } },
          { paid: true, pricing: { output_msat_per_million: 2500 } }
        ]
      }
    },
    { id: 'mixed-model', payment: { free_available: true, paid_available: true, offers: [] } },
    { id: 'no-payment-data' }
  ]
}

describe('model prices', () => {
  const payments = parseModelPayments(listing)

  it('parses the cheapest paid output rate', () => {
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

  it('labels paid listings', () => {
    expect(priceLabel(payments.get('paid-model'))).toBe('from 1.5k msat/M out')
    expect(priceLabel(payments.get('mixed-model'))).toBe('free · paid')
    expect(priceLabel(payments.get('free-model'))).toBeUndefined()
  })
})
