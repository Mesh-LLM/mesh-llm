/** Per-model payment summary from the OpenAI `/v1/models` listing (`payment` block). Descriptive only. */
export type ModelPayment = { freeAvailable: boolean; paidAvailable: boolean; outputMsatPerMillion?: number }

type RawOffer = { paid?: unknown; pricing?: { output_msat_per_million?: unknown } | null }
type RawModel = { id?: unknown; payment?: { free_available?: unknown; paid_available?: unknown; offers?: unknown } }

export function parseModelPayments(body: unknown): Map<string, ModelPayment> {
  const result = new Map<string, ModelPayment>()
  const data = (body as { data?: unknown } | null)?.data
  if (!Array.isArray(data)) return result
  for (const item of data as RawModel[]) {
    if (typeof item?.id !== 'string' || !item.payment) continue
    const offers = Array.isArray(item.payment.offers) ? (item.payment.offers as RawOffer[]) : []
    const rates = offers
      .filter((offer) => offer.paid === true)
      .map((offer) => offer.pricing?.output_msat_per_million)
      .filter((rate): rate is number => typeof rate === 'number')
    result.set(item.id, {
      freeAvailable: item.payment.free_available === true,
      paidAvailable: item.payment.paid_available === true,
      outputMsatPerMillion: rates.length > 0 ? Math.min(...rates) : undefined
    })
  }
  return result
}

/** Free mode hides only models known to be paid-only; models without payment data stay visible. */
export function isOfferedForMode(payment: ModelPayment | undefined, freeOnly: boolean): boolean {
  return !freeOnly || !payment || payment.freeAvailable || !payment.paidAvailable
}

function formatMsat(value: number): string {
  return value >= 1000 ? `${Number((value / 1000).toFixed(1))}k` : String(value)
}

/** Short price label for paid-capable listings, e.g. `1.5k msat/M out` or `free · paid`. */
export function priceLabel(payment: ModelPayment | undefined): string | undefined {
  if (!payment?.paidAvailable) return undefined
  if (payment.freeAvailable) return 'free · paid'
  return payment.outputMsatPerMillion != null ? `${formatMsat(payment.outputMsatPerMillion)} msat/M out` : 'paid'
}
