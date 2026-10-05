// @vitest-environment jsdom

import { renderHook } from '@testing-library/react'
import { afterEach, expect, it, vi } from 'vitest'
import { useAdvancingChartClock } from '@/features/logs/lib/use-advancing-chart-clock'

afterEach(() => {
  vi.restoreAllMocks()
  vi.useRealTimers()
})

it('keeps the initial clock stable until the first aligned minute tick', () => {
  vi.useFakeTimers()
  vi.spyOn(Date, 'now').mockReturnValueOnce(1_000).mockReturnValue(1_001)

  const { result } = renderHook(() => useAdvancingChartClock())

  expect(result.current).toBe(1_000)
})
