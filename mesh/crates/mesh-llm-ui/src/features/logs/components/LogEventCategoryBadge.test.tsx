// @vitest-environment jsdom
import { render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import { LogEventCategoryBadge } from '@/features/logs/components/LogEventCategoryBadge'
import {
  LOG_EVENT_CATEGORY_COLORS,
  LOG_EVENT_CATEGORY_LABELS,
  LOG_EVENT_CATEGORY_MARKER_CLASS
} from '@/features/logs/lib/log-event-category-style'
import { LOG_EVENT_CATEGORIES } from '@/features/logs/lib/log-event-ledger'

describe('LogEventCategoryBadge', () => {
  it.each(LOG_EVENT_CATEGORIES)('labels and tags the %s category', (category) => {
    // Given a ledger row category
    // When the badge renders
    render(<LogEventCategoryBadge category={category} />)

    // Then it carries the shared label and a machine-readable category tag
    const badge = screen.getByText(LOG_EVENT_CATEGORY_LABELS[category])
    expect(badge).toHaveAttribute('data-log-category', category)
  })

  it('paints the marker with the same token the chart legend uses', () => {
    // Given the gossip category
    // When the badge renders
    const { container } = render(<LogEventCategoryBadge category="gossip" />)

    // Then the marker swatch reuses the shared colour token and legend shape
    const marker = container.querySelector('[aria-hidden="true"]')
    expect(marker).toHaveStyle({ backgroundColor: LOG_EVENT_CATEGORY_COLORS.gossip })
    expect(marker).toHaveClass('h-1.5', 'w-2.5')
  })

  it('defines a label, colour, and marker for every ledger category', () => {
    // Given the full ledger category vocabulary
    // When the shared presentation maps are inspected
    // Then no category is missing a visual identity
    for (const category of LOG_EVENT_CATEGORIES) {
      expect(LOG_EVENT_CATEGORY_LABELS[category]).toBeTruthy()
      expect(LOG_EVENT_CATEGORY_COLORS[category]).toBe(`var(--color-log-${category})`)
      expect(LOG_EVENT_CATEGORY_MARKER_CLASS[category]).toBeTruthy()
    }
  })
})
