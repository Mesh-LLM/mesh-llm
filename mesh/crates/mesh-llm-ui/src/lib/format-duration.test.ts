import { describe, it, expect } from 'vitest'
import { formatElapsedMs, formatShortDuration } from '@/lib/format-duration'

const S = 1
const M = 60
const H = 3_600
const D = 86_400
const W = 7 * D
const MO = 30 * D
const Y = 365 * D

describe('formatShortDuration', () => {
  it.each([
    [null, '-'],
    [undefined, '-'],
    [0, '-'],
    [-1, '-'],
    [NaN, '-'],
    [Infinity, '-'],
    [-Infinity, '-']
  ])('returns dash for %s', (input, expected) => {
    expect(formatShortDuration(input as number | null | undefined)).toBe(expected)
  })

  it.each([
    [1, '1s'],
    [20, '20s'],
    [59, '59s']
  ])('formats %i seconds as seconds', (input, expected) => {
    expect(formatShortDuration(input)).toBe(expected)
  })

  it.each([
    [60, '1m'],
    [120, '2m'],
    [3599, '59m']
  ])('formats %i seconds as minutes', (input, expected) => {
    expect(formatShortDuration(input)).toBe(expected)
  })

  it.each([
    [H, '1h'],
    [2 * H, '2h'],
    [23 * H + 59 * M + 59 * S, '23h']
  ])('formats %i seconds as hours', (input, expected) => {
    expect(formatShortDuration(input)).toBe(expected)
  })

  it.each([
    [D, '1d'],
    [2 * D, '2d'],
    [6 * D, '6d']
  ])('formats %i seconds as days', (input, expected) => {
    expect(formatShortDuration(input)).toBe(expected)
  })

  it.each([
    [W, '1w'],
    [2 * W, '2w'],
    [3 * W, '3w']
  ])('formats %i seconds as exact weeks', (input, expected) => {
    expect(formatShortDuration(input)).toBe(expected)
  })

  it.each([
    [9 * D, '1w 2d'],
    [12 * D, '1w 5d'],
    [20 * D, '2w 6d'],
    [27 * D, '3w 6d']
  ])('formats %i seconds as compound weeks+days', (input, expected) => {
    expect(formatShortDuration(input)).toBe(expected)
  })

  it.each([
    [MO, '1mo'],
    [2 * MO, '2mo'],
    [11 * MO, '11mo']
  ])('formats %i seconds as exact months', (input, expected) => {
    expect(formatShortDuration(input)).toBe(expected)
  })

  it.each([
    [MO + 10 * D, '1mo 10d'],
    [2 * MO + 15 * D, '2mo 15d'],
    [3 * MO + D, '3mo 1d']
  ])('formats %i seconds as compound months+days', (input, expected) => {
    expect(formatShortDuration(input)).toBe(expected)
  })

  it.each([
    [Y, '1y'],
    [2 * Y, '2y']
  ])('formats %i seconds as exact years', (input, expected) => {
    expect(formatShortDuration(input)).toBe(expected)
  })

  it.each([
    [Y + 2 * MO, '1y 2mo'],
    [2 * Y + 6 * MO, '2y 6mo']
  ])('formats %i seconds as compound years+months', (input, expected) => {
    expect(formatShortDuration(input)).toBe(expected)
  })

  it.each([
    [Y + 2 * MO + 2 * D, '1y 2mo 2d'],
    [Y + MO + 15 * D, '1y 1mo 15d'],
    [2 * Y + 3 * MO + 7 * D, '2y 3mo 7d']
  ])('formats %i seconds as compound years+months+days', (input, expected) => {
    expect(formatShortDuration(input)).toBe(expected)
  })

  it('floors fractional seconds', () => {
    expect(formatShortDuration(59.9)).toBe('59s')
    expect(formatShortDuration(3601.5)).toBe('1h')
  })
})

const SECOND_MS = 1_000
const MINUTE_MS = 60 * SECOND_MS
const HOUR_MS = 60 * MINUTE_MS

describe('formatElapsedMs', () => {
  it.each([null, undefined, -1, NaN, Infinity])('returns dash for %s', (input) => {
    expect(formatElapsedMs(input)).toBe('-')
  })

  it.each([
    [0, '0ms'],
    [12, '12ms'],
    [388, '388ms'],
    [999, '999ms'],
    [12.4, '12ms']
  ])('formats %s as milliseconds', (input, expected) => {
    expect(formatElapsedMs(input)).toBe(expected)
  })

  it.each([
    [SECOND_MS, '1s'],
    [2 * SECOND_MS, '2s'],
    [2_450, '2.5s'],
    [51_000, '51s'],
    [59_940, '59.9s']
  ])('formats %s as seconds', (input, expected) => {
    expect(formatElapsedMs(input)).toBe(expected)
  })

  it.each([
    [59_950, '1m'],
    [59_999, '1m']
  ])('rounds %s milliseconds up to one minute', (input, expected) => {
    expect(formatElapsedMs(input)).toBe(expected)
  })

  it.each([
    [MINUTE_MS, '1m'],
    [2 * MINUTE_MS + 5 * SECOND_MS, '2m 5s'],
    [59 * MINUTE_MS + 59 * SECOND_MS, '59m 59s']
  ])('formats %s as minutes', (input, expected) => {
    expect(formatElapsedMs(input)).toBe(expected)
  })

  it.each([
    [HOUR_MS, '1h'],
    [2 * HOUR_MS + 30 * MINUTE_MS, '2h 30m']
  ])('formats %s as hours', (input, expected) => {
    expect(formatElapsedMs(input)).toBe(expected)
  })

  it('prefixes the formatted value when a prefix is supplied', () => {
    expect(formatElapsedMs(388, { prefix: '+' })).toBe('+388ms')
    expect(formatElapsedMs(2 * SECOND_MS, { prefix: '+' })).toBe('+2s')
  })

  it('omits the prefix for values it cannot format', () => {
    expect(formatElapsedMs(undefined, { prefix: '+' })).toBe('-')
  })
})
