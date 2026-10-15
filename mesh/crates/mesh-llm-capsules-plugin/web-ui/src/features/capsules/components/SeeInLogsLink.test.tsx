import { cleanup, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it } from 'vitest'
import { SeeInLogsLink } from '@/features/capsules/components/ExchangeIdCell'
import { SEE_IN_LOGS_NOT_ON_THIS_PAGE } from '@/features/capsules/lib/tooltip-copy'

/** A host-minted exchange id (a UUID), the kind a Logs request can carry. */
const EXCHANGE_ID = '0f8fad5b-d9cb-469f-a165-70867728950e'

afterEach(() => {
  cleanup()
})

describe('see in Logs from an exchange row', () => {
  it('says in plain text that this page cannot open Logs there yet: no dead link', () => {
    render(<SeeInLogsLink exchangeKey={EXCHANGE_ID} />)
    expect(screen.getByText(SEE_IN_LOGS_NOT_ON_THIS_PAGE)).toBeInTheDocument()
    expect(screen.queryByRole('button')).not.toBeInTheDocument()
    expect(screen.queryByRole('link')).not.toBeInTheDocument()
  })

  it('shows nothing for a digest-keyed row, which no Logs request carries', () => {
    const { container } = render(<SeeInLogsLink exchangeKey="digest:abc" />)
    expect(container).toBeEmptyDOMElement()
  })
})
