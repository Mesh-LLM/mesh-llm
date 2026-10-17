import { describe, expect, it } from 'vitest'
import { lookupPath, recordPath } from '@/features/capsules/lib/evidence-lookup'

describe('the chat and Logs contributions look an exchange up by the host ids they are given', () => {
  it('a Logs request by its exchange id; a chat answer by its client nonce', () => {
    expect(lookupPath({ slot: 'logs_request', requestId: 'r', exchangeId: 'ex 1' })).toBe('http/lookup?exchange_id=ex%201')
    expect(lookupPath({ slot: 'chat_message', messageId: 'm', clientNonce: 'n/1' })).toBe('http/lookup?client_nonce=n%2F1')
  })

  it('no id from the host: no lookup at all, nothing guessed', () => {
    expect(lookupPath({ slot: 'logs_request', requestId: 'r' })).toBeNull()
    expect(lookupPath({ slot: 'chat_message', messageId: 'm' })).toBeNull()
  })

  it('a found record links to its row on the Evidence page; none, no link', () => {
    expect(recordPath({ found: true, exchange_key: 'digest:ab', capsule_id: 'c' })).toBe(
      '/plugins/capsules/evidence?focusExchangeKey=digest%3Aab'
    )
    expect(recordPath({ found: false })).toBeNull()
    expect(recordPath({ found: true, exchange_key: null, capsule_id: 'c' })).toBeNull()
  })
})
