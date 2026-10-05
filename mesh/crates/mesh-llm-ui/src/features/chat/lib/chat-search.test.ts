import { describe, expect, it } from 'vitest'
import { parseChatSearch } from '@/features/chat/lib/chat-search'

const NODE = 'a70d3967bea3b22fa48a28f77c5d2b3764fc8bd5204a82c09ff8430f3f2a0a00'

describe('parseChatSearch', () => {
  it('keeps a node endpoint id as the target, lowercased', () => {
    expect(parseChatSearch({ target: NODE })).toEqual({ target: NODE })
    expect(parseChatSearch({ target: ` ${NODE.toUpperCase()} ` })).toEqual({ target: NODE })
  })

  it.each([undefined, '', 'carrack', NODE.slice(1), `${NODE}00`, 42])(
    'drops a target that is not an endpoint id: %s',
    (target) => {
      expect(parseChatSearch({ target })).toEqual({})
    }
  )
})
