import { describe, expect, it } from 'vitest'
import { GRADING_WORDS, STANDING_WORD } from '@/features/capsules/lib/banned-terms'
import {
  NOT_ADJUDICATED_LINES,
  NOT_ADJUDICATED_YET,
  notAdjudicatedLine
} from '@/features/capsules/lib/referee-row'

describe('the not-adjudicated reason lines', () => {
  it('a host with no twin bracket ids says so, word for word', () => {
    expect(notAdjudicatedLine({ state: 'not_adjudicated', reason: 'host_does_not_mark_twins' })).toBe(
      'Not adjudicated: no client marked these as a pair.'
    )
  })

  it('every reason the plugin gives has its own line', () => {
    for (const reason of [
      'off',
      'host_does_not_mark_twins',
      'twins_agree',
      'not_comparable',
      'no_eligible_referee',
      'referee_cannot_sign',
      'referee_unreachable'
    ]) {
      expect(notAdjudicatedLine({ state: 'not_adjudicated', reason })).toBe(NOT_ADJUDICATED_LINES[reason])
    }
    expect(new Set(Object.values(NOT_ADJUDICATED_LINES)).size).toBe(Object.keys(NOT_ADJUDICATED_LINES).length)
  })

  it('not comparable says why', () => {
    for (const because of ['sampled', 'model_hash_differs', 'model_hash_unknown', 'weights_differ', 'weights_unknown']) {
      expect(notAdjudicatedLine({ state: 'not_adjudicated', reason: 'not_comparable', because })).toMatch(
        /^Not adjudicated: not comparable, .+\.$/
      )
    }
  })

  it('an adjudicated pair has no such line; a pair with no outcome yet says so', () => {
    expect(
      notAdjudicatedLine({ state: 'adjudicated', verdict: 'contradicted:node-b', referee: 'node-c', tier: 1 })
    ).toBeNull()
    expect(notAdjudicatedLine(null)).toBe(NOT_ADJUDICATED_YET)
  })

  it('no line grades a machine', () => {
    const graded = new RegExp([STANDING_WORD, ...GRADING_WORDS].join('|'), 'i')
    for (const line of [...Object.values(NOT_ADJUDICATED_LINES), NOT_ADJUDICATED_YET]) {
      expect(line).not.toMatch(graded)
    }
  })
})
