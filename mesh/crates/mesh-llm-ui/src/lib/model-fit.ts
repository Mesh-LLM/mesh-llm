export type FitLabelTone = 'good' | 'info' | 'warn' | 'bad' | 'neutral'

export function headerFitLabel(label?: string) {
  if (label === 'Likely comfortable' || label === 'Likely fits') {
    return 'Suitable for this node'
  }
  if (label === 'Possible with tradeoffs') {
    return 'May fit this node'
  }
  if (label === 'Likely too large') {
    return 'Too large for this node'
  }
  return 'Check fit'
}

export function fitLabelTone(label?: string): FitLabelTone {
  if (label === 'Likely comfortable') return 'good'
  if (label === 'Likely fits') return 'good'
  if (label === 'Possible with tradeoffs') return 'warn'
  if (label === 'Likely too large') return 'bad'
  return 'neutral'
}

export function fitLabelTooltip(label?: string) {
  if (label === 'Likely comfortable') {
    return 'Should run comfortably on this node.'
  }
  if (label === 'Likely fits') {
    return 'Should fit on this node, but tightly.'
  }
  if (label === 'Possible with tradeoffs') {
    return 'May fit, with tighter memory or performance tradeoffs.'
  }
  if (label === 'Likely too large') {
    return 'Likely too large for this node alone.'
  }
  return 'Estimated fit for this node.'
}
