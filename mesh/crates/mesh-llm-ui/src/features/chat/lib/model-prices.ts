/** Per-model payment summary from the `/mesh/v1/models` discovery listing (per-provider `payment`). Descriptive only. */
export type ModelPayment = { freeAvailable: boolean; paidAvailable: boolean; outputMsatPerMillion?: number }

/** A provider entry from `/mesh/v1/models`; discovery keeps offline providers listed with `available: false`. */
type RawPayment = { paid?: unknown; pricing?: { output_msat_per_million?: unknown } | null }
type RawProvider = { local?: unknown; available?: unknown; payment?: RawPayment | null }
type RawModel = { id?: unknown; providers?: unknown }

export function parseModelPayments(body: unknown): Map<string, ModelPayment> {
  const result = new Map<string, ModelPayment>()
  const data = (body as { data?: unknown } | null)?.data
  if (!Array.isArray(data)) return result
  for (const item of data as RawModel[]) {
    if (typeof item?.id !== 'string') continue
    // Availability gates every derived field: an offline free or local provider
    // must not make the model free-available, and offline paid rates must not
    // set the shown minimum.
    const providers = (Array.isArray(item.providers) ? (item.providers as RawProvider[]) : []).filter(
      (provider) => provider?.available !== false
    )
    if (!providers.some((provider) => provider?.payment != null)) continue
    const rates = providers
      .map((provider) => provider.payment)
      .filter((payment): payment is RawPayment => payment != null && payment.paid === true)
      .map((payment) => payment.pricing?.output_msat_per_million)
      .filter((rate): rate is number => typeof rate === 'number')
    const payment = {
      // A local provider is free for this node even when it advertises paid
      // offers, and an advertised free offer keeps the model free elsewhere.
      freeAvailable: providers.some((provider) => provider?.local === true || provider?.payment?.paid === false),
      paidAvailable: providers.some((provider) => provider?.payment?.paid === true),
      outputMsatPerMillion: rates.length > 0 ? Math.min(...rates) : undefined
    }
    // Discovery IDs are the public request identities, profile suffixes included.
    result.set(item.id, payment)
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

/** Short price label for paid-capable listings, e.g. `from 1.5k msat/M out` or `free · paid`. */
export function priceLabel(payment: ModelPayment | undefined): string | undefined {
  if (!payment?.paidAvailable) return undefined
  if (payment.freeAvailable) return 'free · paid'
  return payment.outputMsatPerMillion != null ? `from ${formatMsat(payment.outputMsatPerMillion)} msat/M out` : 'paid'
}
