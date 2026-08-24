import { useCallback, useEffect, type Dispatch, type MutableRefObject, type SetStateAction } from 'react'
import { chooseClusteredMeshNodePosition } from '@/features/network/lib/mesh-placement'
import { nextMeshVizDotColorSchemeIndex, themeFromDocument } from '@/features/network/lib/mesh-viz-dot-color-schemes'
import type { MeshNode, ResolvedTheme } from '@/features/app-tabs/types'
import type { MeshVizGridMode } from '@/features/network/components/MeshVizDebugControls'
import {
  createDebugNode,
  debugNodeMatchesShortcut,
  debugNodeShortcutCount,
  getDebugNodeShortcutBlueprint,
  isTextEditingTarget,
  type DebugMeshNode,
  type DebugNodeShortcut
} from '@/features/network/components/MeshViz.helpers'

type UseMeshVizDebugControlsArgs = {
  nodes: MeshNode[]
  meshSeed: string
  debugNodeCounterRef: MutableRefObject<number>
  setDebugNodes: Dispatch<SetStateAction<DebugMeshNode[]>>
  setDotColorSchemeIndex: Dispatch<SetStateAction<number>>
  setDotColorSchemeTheme: Dispatch<SetStateAction<ResolvedTheme>>
  setShowPanBounds: Dispatch<SetStateAction<boolean>>
  setGridMode: Dispatch<SetStateAction<MeshVizGridMode>>
  debugShortcutsEnabled: boolean
  playRandomTraffic: () => void
  playSelfTraffic: () => void
}

export function useMeshVizDebugControls({
  nodes,
  meshSeed,
  debugNodeCounterRef,
  setDebugNodes,
  setDotColorSchemeIndex,
  setDotColorSchemeTheme,
  setShowPanBounds,
  setGridMode,
  debugShortcutsEnabled,
  playRandomTraffic,
  playSelfTraffic
}: UseMeshVizDebugControlsArgs) {
  const addDebugNode = useCallback(
    (shortcut: DebugNodeShortcut) => {
      const blueprint = getDebugNodeShortcutBlueprint(shortcut)
      const debugIndex = debugNodeCounterRef.current + 1
      debugNodeCounterRef.current = debugIndex

      setDebugNodes((current) => {
        const placementNodes: MeshNode[] = [...nodes, ...current]
        const position = chooseClusteredMeshNodePosition(meshSeed, debugIndex, blueprint, placementNodes)
        const debugNode = createDebugNode(debugIndex, blueprint, position)

        return [...current, debugNode]
      })
    },
    [debugNodeCounterRef, meshSeed, nodes, setDebugNodes]
  )

  const removeDebugNode = useCallback(
    (shortcut: DebugNodeShortcut) => {
      setDebugNodes((current) => {
        let removeIndex = -1

        for (let index = current.length - 1; index >= 0; index -= 1) {
          if (debugNodeMatchesShortcut(current[index], shortcut)) {
            removeIndex = index
            break
          }
        }

        if (removeIndex === -1) {
          return current
        }

        return current.filter((_, index) => index !== removeIndex)
      })
    },
    [setDebugNodes]
  )

  const cycleDotColorScheme = useCallback(() => {
    setDotColorSchemeIndex(nextMeshVizDotColorSchemeIndex)
  }, [setDotColorSchemeIndex])

  const selectDotColorScheme = useCallback(
    (index: number) => {
      setDotColorSchemeIndex(index)
    },
    [setDotColorSchemeIndex]
  )

  useEffect(() => {
    if (typeof document === 'undefined' || typeof MutationObserver === 'undefined') {
      return undefined
    }

    const root = document.documentElement
    const syncTheme = () => setDotColorSchemeTheme(themeFromDocument())

    syncTheme()

    const observer = new MutationObserver(syncTheme)
    observer.observe(root, { attributes: true, attributeFilter: ['data-theme'] })

    return () => observer.disconnect()
  }, [setDotColorSchemeTheme])

  useEffect(() => {
    if (!debugShortcutsEnabled) {
      return undefined
    }

    const onKeyDown = (event: KeyboardEvent) => {
      if (event.defaultPrevented || isTextEditingTarget(event.target)) {
        return
      }

      const key = event.key.toLowerCase()
      const debugNodeShortcut = debugNodeShortcutCount(event)

      if (!event.ctrlKey && !event.metaKey && !event.altKey && !event.shiftKey) {
        if (key === 'z') {
          event.preventDefault()
          playRandomTraffic()
          return
        }

        if (key === 'x') {
          event.preventDefault()
          playSelfTraffic()
          return
        }
      }

      if (debugNodeShortcut !== undefined) {
        if (event.shiftKey && !event.ctrlKey && !event.metaKey && !event.altKey) {
          event.preventDefault()
          removeDebugNode(debugNodeShortcut)
          return
        }

        if (!event.ctrlKey || event.metaKey || event.altKey || event.shiftKey) {
          return
        }

        event.preventDefault()
        addDebugNode(debugNodeShortcut)
        return
      }

      if (key === 'b') {
        if (!event.ctrlKey || event.metaKey || event.altKey || event.shiftKey) {
          return
        }

        event.preventDefault()
        setShowPanBounds((current) => !current)
        return
      }

      if (key === 'g') {
        if (!event.ctrlKey || event.metaKey || event.altKey || event.shiftKey) {
          return
        }

        event.preventDefault()
        setGridMode((current) => (current === 'line' ? 'dot' : 'line'))
        return
      }

      if (key === 'c') {
        if (!event.ctrlKey || event.metaKey || event.altKey || event.shiftKey) {
          return
        }

        event.preventDefault()
        cycleDotColorScheme()
      }
    }

    window.addEventListener('keydown', onKeyDown)

    return () => window.removeEventListener('keydown', onKeyDown)
  }, [
    addDebugNode,
    cycleDotColorScheme,
    debugShortcutsEnabled,
    playRandomTraffic,
    playSelfTraffic,
    removeDebugNode,
    setGridMode,
    setShowPanBounds
  ])

  return {
    addDebugNode,
    cycleDotColorScheme,
    removeDebugNode,
    selectDotColorScheme
  }
}
