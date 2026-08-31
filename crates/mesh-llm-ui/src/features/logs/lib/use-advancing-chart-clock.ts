import { useEffect, useRef, useState } from 'react'

const CLOCK_TICK_MS = 60_000

export function useAdvancingChartClock(enabled = true): number {
  const [current, setCurrent] = useState(() => Date.now())
  const wasEnabled = useRef(enabled)

  useEffect(() => {
    if (!enabled) {
      wasEnabled.current = false
      return
    }

    if (!wasEnabled.current) {
      // Re-enabling must re-anchor the rolling window before its aligned timer starts.
      setCurrent(Date.now())
    }
    wasEnabled.current = true
    let interval: number | undefined
    const delay = CLOCK_TICK_MS - (Date.now() % CLOCK_TICK_MS)
    const timeout = window.setTimeout(() => {
      setCurrent(Date.now())
      interval = window.setInterval(() => setCurrent(Date.now()), CLOCK_TICK_MS)
    }, delay)

    return () => {
      window.clearTimeout(timeout)
      if (interval !== undefined) window.clearInterval(interval)
    }
  }, [enabled])

  return current
}
