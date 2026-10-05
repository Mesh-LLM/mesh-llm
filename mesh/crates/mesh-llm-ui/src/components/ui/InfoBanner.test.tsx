import '@testing-library/jest-dom/vitest'

import { render, screen, within } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import { InfoBanner } from '@/components/ui/InfoBanner'

describe('InfoBanner', () => {
  it('keeps the default banner as a named region and renders its annotation', () => {
    render(
      <InfoBanner
        annotation={<span>Window notice</span>}
        description="Current activity remains available."
        title="System activity"
        titleId="system-activity-title"
      />
    )

    const region = screen.getByRole('region', { name: 'System activity' })
    expect(within(region).getByText('Window notice')).toBeVisible()
    expect(screen.queryByRole('alert')).not.toBeInTheDocument()
  })

  it('announces the error variant once while keeping annotation details visible', () => {
    render(
      <InfoBanner
        annotation={<span>Request history: Unavailable</span>}
        description="Request history could not be loaded."
        title="System logs"
        titleId="system-logs-title"
        variant="error"
      />
    )

    const alert = screen.getByRole('alert', { name: 'System logs' })
    expect(alert).toHaveTextContent('Request history could not be loaded.')
    expect(within(alert).getByText('Request history: Unavailable')).toBeVisible()
    expect(screen.getAllByRole('alert')).toHaveLength(1)
  })
})
