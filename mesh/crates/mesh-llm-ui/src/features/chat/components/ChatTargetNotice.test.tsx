import '@testing-library/jest-dom/vitest'

import { fireEvent, render, screen } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'
import { ChatTargetNotice } from '@/features/chat/components/ChatTargetNotice'

const NODE = 'a70d3967bea3b22fa48a28f77c5d2b3764fc8bd5204a82c09ff8430f3f2a0a00'

describe('ChatTargetNotice', () => {
  it('names the target node and clears it', () => {
    const onClear = vi.fn()
    render(<ChatTargetNotice target={NODE} onClear={onClear} />)

    expect(screen.getByRole('status')).toHaveTextContent('Sending to a70d3967bea3')
    expect(screen.getByTitle(NODE)).toBeInTheDocument()
    fireEvent.click(screen.getByRole('button', { name: 'clear' }))
    expect(onClear).toHaveBeenCalledTimes(1)
  })
})
