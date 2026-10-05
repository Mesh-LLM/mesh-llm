import '@testing-library/jest-dom/vitest'

import { render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import { LogRequestTimeline, type LogRequestTimelineProps } from '@/features/logs/components/LogRequestTimeline'

const READY_TIMELINE: LogRequestTimelineProps = {
  events: [],
  attempts: [],
  eventsLoading: false,
  eventsError: false,
  attemptsLoading: false,
  attemptsError: false
}

const queryStates = [
  {
    name: 'loading',
    role: 'status',
    props: { eventsLoading: true, attemptsLoading: true }
  },
  {
    name: 'error',
    role: 'alert',
    props: { eventsError: true, attemptsError: true }
  }
] as const

describe('LogRequestTimeline', () => {
  it.each(queryStates)('keeps $name notices compact and wrap-safe', ({ role, props }) => {
    // Given / When
    const view = render(<LogRequestTimeline {...READY_TIMELINE} {...props} />)

    // Then
    expect(view.container.firstElementChild).toHaveClass('min-w-0', 'gap-[var(--shell-normal)]')
    const notices = screen.getAllByRole(role)
    expect(notices).toHaveLength(2)
    for (const notice of notices) {
      expect(notice).toHaveClass('min-w-0', 'break-words', 'px-[var(--panel-x)]', 'py-[var(--panel-y)]')
    }
  })
})
