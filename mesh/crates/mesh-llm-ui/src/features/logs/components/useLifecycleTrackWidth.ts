import { useCallback, useEffect, useRef, useState } from 'react'

export function useTrackWidth() {
  const [width, setWidth] = useState(0)
  const observerRef = useRef<ResizeObserver | null>(null)

  const trackRef = useCallback((node: HTMLDivElement | null) => {
    observerRef.current?.disconnect()
    observerRef.current = null
    if (!node) return
    const observer = new ResizeObserver((entries) => {
      const entry = entries[0]
      if (entry) setWidth(entry.contentRect.width)
    })
    observer.observe(node)
    observerRef.current = observer
    setWidth(node.getBoundingClientRect().width)
  }, [])

  useEffect(() => () => observerRef.current?.disconnect(), [])

  return { trackRef, width }
}
