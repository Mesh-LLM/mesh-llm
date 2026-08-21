import '@testing-library/jest-dom/vitest'

import { render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import type { LogRequest } from '@/features/logs/api/schemas'
import { LogRequestOverviewMetadata } from '@/features/logs/components/LogRequestOverviewMetadata'
import { HARNESS_LOG_FIXTURES, HARNESS_LOG_SCENARIO_IDS } from '@/features/logs/lib/log-fixtures'

function requestFixture(requestId: string): LogRequest {
  const request = HARNESS_LOG_FIXTURES.find((candidate) => candidate.requestId.toString() === requestId)
  if (request === undefined) throw new Error(`Missing request fixture ${requestId}`)
  return request
}

describe('LogRequestOverviewMetadata', () => {
  it('resets the trailing request-metadata field to a single lg column instead of inheriting the sm span', () => {
    const request = requestFixture(HARNESS_LOG_SCENARIO_IDS.completedMesh.toString())
    render(
      <LogRequestOverviewMetadata artifacts={{ items: undefined, loading: false, error: false }} request={request} />
    )

    // Nine request-metadata fields: the trailing field needs sm:col-span-2 to
    // fill the last (odd) row of the 2-column layout, but the 3-column layout
    // divides evenly and must reset it back to a single lg column rather than
    // inheriting the sm span and leaving an empty cell.
    const recordSource = screen.getByText('Record source').closest('div')
    expect(recordSource).toHaveClass('sm:col-span-2')
    expect(recordSource).toHaveClass('lg:col-span-1')
  })
})
