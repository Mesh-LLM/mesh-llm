import { fireEvent, render, screen } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'
import { Composer } from './Composer'

describe('Composer request eligibility', () => {
  it('blocks sending and retry but preserves Stop and Escape for an active response', () => {
    const onStop = vi.fn()
    const onSend = vi.fn()
    render(
      <Composer
        value="Next prompt"
        onChange={vi.fn()}
        onSend={onSend}
        onStop={onStop}
        onRetry={vi.fn()}
        canRetry
        isStreaming
        requestDisabled
      />
    )
    expect(screen.getByRole('button', { name: 'Stop' })).toBeEnabled()
    expect(screen.getByLabelText('Prompt')).toBeEnabled()
    fireEvent.keyDown(screen.getByLabelText('Prompt'), { key: 'Enter' })
    expect(onSend).not.toHaveBeenCalled()
    fireEvent.click(screen.getByRole('button', { name: 'Stop' }))
    fireEvent.keyDown(screen.getByLabelText('Prompt'), { key: 'Escape' })
    expect(onStop).toHaveBeenCalledTimes(2)
  })
})
