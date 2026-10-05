import '@testing-library/jest-dom/vitest'

import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { CopyInstructionRow } from '@/components/ui/CopyInstructionRow'

function installClipboard(writeText: (text: string) => Promise<void>) {
  Object.defineProperty(navigator, 'clipboard', {
    configurable: true,
    value: { writeText }
  })
}

describe('CopyInstructionRow', () => {
  afterEach(() => {
    Object.defineProperty(navigator, 'clipboard', { configurable: true, value: undefined })
  })

  it('uses the shared responsive control while copying the configured value', async () => {
    const user = userEvent.setup()
    const writeText = vi.fn<(text: string) => Promise<void>>().mockResolvedValue(undefined)
    installClipboard(writeText)
    render(<CopyInstructionRow copyValue="request-id" label="Request ID" value="visible-id" />)

    const copyControl = screen.getByRole('button', { name: 'Copy Request ID' })
    expect(copyControl).toHaveClass(
      'h-13',
      'min-h-11',
      'lg:h-8',
      'lg:min-h-8',
      'focus-visible:!outline-2',
      'focus-visible:!outline-accent',
      'focus-visible:!outline-solid',
      'focus-visible:ring-2',
      'focus-visible:ring-ring'
    )

    await user.click(copyControl)

    expect(writeText).toHaveBeenCalledWith('request-id')
    expect(await screen.findByText('Copied')).toBeInTheDocument()
  })

  it('keeps unavailable copy controls disabled without writing to the clipboard', async () => {
    const user = userEvent.setup()
    const writeText = vi.fn<(text: string) => Promise<void>>().mockResolvedValue(undefined)
    installClipboard(writeText)
    render(<CopyInstructionRow disabled label="Request ID" value="request-id" />)

    const copyControl = screen.getByRole('button', { name: 'Copy Request ID' })
    expect(copyControl).toBeDisabled()
    expect(copyControl).toHaveTextContent('Unavailable')

    await user.click(copyControl)

    expect(writeText).not.toHaveBeenCalled()
  })
})
