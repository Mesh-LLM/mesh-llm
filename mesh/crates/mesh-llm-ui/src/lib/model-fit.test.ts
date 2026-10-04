import { describe, expect, it } from 'vitest'
import { fitLabelTone, fitLabelTooltip, headerFitLabel } from '@/lib/model-fit'

describe('headerFitLabel', () => {
  it.each([
    ['Likely comfortable', 'Suitable for this node'],
    ['Likely fits', 'Suitable for this node'],
    ['Possible with tradeoffs', 'May fit this node'],
    ['Likely too large', 'Too large for this node'],
    ['Unknown', 'Check fit'],
    [undefined, 'Check fit']
  ])('maps %j to %j', (label, expected) => {
    expect(headerFitLabel(label)).toBe(expected)
  })
})

describe('fitLabelTone', () => {
  it.each([
    ['Likely comfortable', 'good'],
    ['Likely fits', 'good'],
    ['Possible with tradeoffs', 'warn'],
    ['Likely too large', 'bad'],
    ['Unknown', 'neutral'],
    [undefined, 'neutral']
  ])('maps %j to %j', (label, expected) => {
    expect(fitLabelTone(label)).toBe(expected)
  })
})

describe('fitLabelTooltip', () => {
  it('returns a specific tooltip for known labels and a generic one otherwise', () => {
    expect(fitLabelTooltip('Likely comfortable')).toBe('Should run comfortably on this node.')
    expect(fitLabelTooltip('Likely too large')).toBe('Likely too large for this node alone.')
    expect(fitLabelTooltip('Unknown')).toBe('Estimated fit for this node.')
    expect(fitLabelTooltip(undefined)).toBe('Estimated fit for this node.')
  })
})
