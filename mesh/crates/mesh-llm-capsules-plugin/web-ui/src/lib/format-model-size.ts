export function formatModelSizeGB(value: number | null | undefined): string {
  if (value == null || !Number.isFinite(value) || value <= 0) {
    return 'Unknown'
  }

  const roundedToTenths = Math.round(value * 10) / 10

  if (Number.isInteger(roundedToTenths)) {
    return `${roundedToTenths} GB`
  }

  return `${roundedToTenths.toFixed(1)} GB`
}
