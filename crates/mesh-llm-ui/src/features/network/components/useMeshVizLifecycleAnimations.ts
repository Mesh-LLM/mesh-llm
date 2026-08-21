import { useEffect, useLayoutEffect, type Dispatch, type MutableRefObject, type SetStateAction } from 'react'
import { createTimeline, type Timeline } from 'animejs'
import type { MeshNode } from '@/features/app-tabs/types'
import type { MeshVizNodeLifecycle } from '@/features/network/components/MeshVizNode'
type MeshVizLink = {
  readonly id: string
  readonly source: Pick<MeshNode, 'id'>
  readonly target: Pick<MeshNode, 'id'>
}

export type MeshLifecycleTimelineRecord = {
  keys: Set<string>
  timeline: Timeline
}

export type LinkRestoreTimelineRecord = {
  linkIds: Set<string>
  timeline: Timeline
}

type UseMeshVizLifecycleAnimationsArgs = {
  currentRenderNodes: MeshNode[]
  currentRenderNodeIds: Set<string>
  renderNodes: MeshNode[]
  links: readonly MeshVizLink[]
  reduceMotion: boolean
  nodeLifecyclePhase: (nodeId: string) => MeshVizNodeLifecycle
  linkLifecyclePhase: (sourceNodeId: string, targetNodeId: string) => MeshVizNodeLifecycle
  nodeLayerRef: MutableRefObject<HTMLDivElement | null>
  svgPanLayerRef: MutableRefObject<SVGGElement | null>
  topologyNodeSnapshotsRef: MutableRefObject<Map<string, MeshNode>>
  hasTopologySnapshotRef: MutableRefObject<boolean>
  nodeLifecycleTimeoutsRef: MutableRefObject<Map<string, number>>
  nodeLifecycleAnimationKeysRef: MutableRefObject<Set<string>>
  meshLifecycleTimelineRecordsRef: MutableRefObject<Map<number, MeshLifecycleTimelineRecord>>
  meshLifecycleTimelineIdRef: MutableRefObject<number>
  previousLinkIdsRef: MutableRefObject<Set<string>>
  linkRestoreAnimationIdsRef: MutableRefObject<Set<string>>
  linkRestoreTimelineRecordsRef: MutableRefObject<Map<number, LinkRestoreTimelineRecord>>
  linkRestoreTimelineIdRef: MutableRefObject<number>
  setNodeLifecyclePhases: Dispatch<SetStateAction<Record<string, MeshVizNodeLifecycle>>>
  setExitingNodes: Dispatch<SetStateAction<MeshNode[]>>
  openNodeId: string | undefined
  localHoveredNodeId: string | undefined
  setOpenNodeId: Dispatch<SetStateAction<string | undefined>>
  setLocalHoveredNodeId: Dispatch<SetStateAction<string | undefined>>
}
const NODE_JOIN_STAGGER_MS = 720
const NODE_JOIN_DURATION_MS = 380
const LINK_JOIN_DURATION_MS = 420
const CONNECTED_NODE_PULSE_DELAY_MS = 500
const CONNECTED_NODE_PULSE_DURATION_MS = 360
const NODE_JOIN_SETTLE_BUFFER_MS = 60
const LINK_LEAVE_DURATION_MS = 260
const NODE_LEAVE_DELAY_MS = 90
const NODE_LEAVE_DURATION_MS = 280
const NODE_LEAVE_STAGGER_MS = 260
const NODE_LEAVE_SETTLE_BUFFER_MS = 120

function lifecycleTransitionKey(nodeId: string, phase: MeshVizNodeLifecycle) {
  return `${nodeId}\u001f${phase}`
}

function isDefined<T>(value: T | undefined): value is T {
  return value !== undefined
}

function numericCssValue(value: string | null | undefined, fallback: number) {
  if (!value || value === 'none') {
    return fallback
  }

  const parsedValue = Number.parseFloat(value)
  return Number.isFinite(parsedValue) ? parsedValue : fallback
}

function currentElementOpacity(element: Element, fallback: number) {
  return numericCssValue(
    element instanceof HTMLElement || element instanceof SVGElement
      ? element.style.opacity || element.getAttribute('opacity') || window.getComputedStyle(element).opacity
      : element.getAttribute('opacity'),
    fallback
  )
}

function currentElementScale(element: HTMLElement, fallback: number) {
  return numericCssValue(element.style.scale || window.getComputedStyle(element).scale, fallback)
}

function currentStrokeDashOffset(element: SVGElement, fallback: number) {
  return numericCssValue(
    element.style.getPropertyValue('stroke-dashoffset') ||
      element.getAttribute('stroke-dashoffset') ||
      window.getComputedStyle(element).getPropertyValue('stroke-dashoffset'),
    fallback
  )
}

function nodeJoinSettleDelay(index: number) {
  return (
    index * NODE_JOIN_STAGGER_MS +
    CONNECTED_NODE_PULSE_DELAY_MS +
    CONNECTED_NODE_PULSE_DURATION_MS +
    NODE_JOIN_SETTLE_BUFFER_MS
  )
}

function nodeLeaveRemovalDelay(index: number) {
  return index * NODE_LEAVE_STAGGER_MS + NODE_LEAVE_DELAY_MS + NODE_LEAVE_DURATION_MS + NODE_LEAVE_SETTLE_BUFFER_MS
}
export function useMeshVizLifecycleAnimations({
  currentRenderNodes,
  currentRenderNodeIds,
  renderNodes,
  links,
  reduceMotion,
  nodeLifecyclePhase,
  linkLifecyclePhase,
  nodeLayerRef,
  svgPanLayerRef,
  topologyNodeSnapshotsRef,
  hasTopologySnapshotRef,
  nodeLifecycleTimeoutsRef,
  nodeLifecycleAnimationKeysRef,
  meshLifecycleTimelineRecordsRef,
  meshLifecycleTimelineIdRef,
  previousLinkIdsRef,
  linkRestoreAnimationIdsRef,
  linkRestoreTimelineRecordsRef,
  linkRestoreTimelineIdRef,
  setNodeLifecyclePhases,
  setExitingNodes,
  openNodeId,
  localHoveredNodeId,
  setOpenNodeId,
  setLocalHoveredNodeId
}: UseMeshVizLifecycleAnimationsArgs) {
  useEffect(() => {
    const previousSnapshots = topologyNodeSnapshotsRef.current
    const hasTopologySnapshot = hasTopologySnapshotRef.current
    const currentIds = new Set(currentRenderNodes.map((node) => node.id))
    const addedNodes = hasTopologySnapshot ? currentRenderNodes.filter((node) => !previousSnapshots.has(node.id)) : []
    const removedNodes = [...previousSnapshots.entries()]
      .filter(([nodeId]) => !currentIds.has(nodeId))
      .map(([, node]) => node)

    if (addedNodes.length > 0 || removedNodes.length > 0) {
      setNodeLifecyclePhases((current) => {
        const next = { ...current }

        for (const node of addedNodes) {
          next[node.id] = reduceMotion ? 'present' : 'entering'
        }

        for (const node of removedNodes) {
          if (reduceMotion) {
            delete next[node.id]
          } else {
            next[node.id] = 'leaving'
          }
        }

        return next
      })
    }

    if (reduceMotion) {
      for (const node of [...addedNodes, ...removedNodes]) {
        const existingTimeout = nodeLifecycleTimeoutsRef.current.get(node.id)

        if (existingTimeout !== undefined) {
          window.clearTimeout(existingTimeout)
          nodeLifecycleTimeoutsRef.current.delete(node.id)
        }
      }

      if (removedNodes.length > 0) {
        const removedNodeIds = new Set(removedNodes.map((node) => node.id))
        setExitingNodes((current) => current.filter((node) => !removedNodeIds.has(node.id)))
      }

      topologyNodeSnapshotsRef.current = new Map(currentRenderNodes.map((node) => [node.id, node]))
      hasTopologySnapshotRef.current = true
      return
    }

    if (addedNodes.length > 0) {
      setExitingNodes((current) => current.filter((node) => !currentIds.has(node.id)))

      addedNodes.forEach((node, index) => {
        const existingTimeout = nodeLifecycleTimeoutsRef.current.get(node.id)

        if (existingTimeout !== undefined) {
          window.clearTimeout(existingTimeout)
        }

        const timeout = window.setTimeout(() => {
          nodeLifecycleTimeoutsRef.current.delete(node.id)
          setNodeLifecyclePhases((current) => {
            if (current[node.id] !== 'entering') return current
            return { ...current, [node.id]: 'present' }
          })
        }, nodeJoinSettleDelay(index))

        nodeLifecycleTimeoutsRef.current.set(node.id, timeout)
      })
    }

    if (removedNodes.length > 0) {
      setExitingNodes((current) => {
        const activeExitingNodes = current.filter((node) => !currentIds.has(node.id))
        const activeExitingIds = new Set(activeExitingNodes.map((node) => node.id))
        const nextRemovedNodes = removedNodes.filter((node) => !activeExitingIds.has(node.id))

        return [...activeExitingNodes, ...nextRemovedNodes]
      })

      removedNodes.forEach((node, index) => {
        const existingTimeout = nodeLifecycleTimeoutsRef.current.get(node.id)

        if (existingTimeout !== undefined) {
          window.clearTimeout(existingTimeout)
        }

        const timeout = window.setTimeout(() => {
          nodeLifecycleTimeoutsRef.current.delete(node.id)
          setExitingNodes((current) => current.filter((exitingNode) => exitingNode.id !== node.id))
          setNodeLifecyclePhases((current) => {
            if (!(node.id in current)) return current

            const { [node.id]: removedPhase, ...next } = current
            void removedPhase
            return next
          })
        }, nodeLeaveRemovalDelay(index))

        nodeLifecycleTimeoutsRef.current.set(node.id, timeout)
      })
    }

    topologyNodeSnapshotsRef.current = new Map(currentRenderNodes.map((node) => [node.id, node]))
    hasTopologySnapshotRef.current = true
  }, [
    currentRenderNodes,
    hasTopologySnapshotRef,
    nodeLifecycleTimeoutsRef,
    reduceMotion,
    setExitingNodes,
    setNodeLifecyclePhases,
    topologyNodeSnapshotsRef
  ])

  useLayoutEffect(() => {
    const activeTransitionKeys = new Set<string>()
    const transitioningNodes = renderNodes
      .map((node) => ({ node, phase: nodeLifecyclePhase(node.id) }))
      .filter(({ phase }) => phase === 'entering' || phase === 'leaving')

    for (const { node, phase } of transitioningNodes) {
      activeTransitionKeys.add(lifecycleTransitionKey(node.id, phase))
    }

    for (const key of [...nodeLifecycleAnimationKeysRef.current]) {
      if (!activeTransitionKeys.has(key)) {
        nodeLifecycleAnimationKeysRef.current.delete(key)
      }
    }

    for (const [recordId, record] of meshLifecycleTimelineRecordsRef.current) {
      const stillActive = [...record.keys].some((key) => activeTransitionKeys.has(key))

      if (!stillActive) {
        record.timeline.revert()
        meshLifecycleTimelineRecordsRef.current.delete(recordId)
      }
    }

    if (reduceMotion) {
      for (const record of meshLifecycleTimelineRecordsRef.current.values()) {
        record.timeline.revert()
      }

      meshLifecycleTimelineRecordsRef.current.clear()
      nodeLifecycleAnimationKeysRef.current.clear()
      return undefined
    }

    if (transitioningNodes.length === 0) {
      return undefined
    }

    const nodeLayerElement = nodeLayerRef.current
    const svgPanLayerElement = svgPanLayerRef.current

    if (!nodeLayerElement || !svgPanLayerElement) {
      return undefined
    }

    const nodeCoreElements = new Map<string, HTMLElement>()
    nodeLayerElement.querySelectorAll<HTMLElement>('[data-mesh-node-core]').forEach((element) => {
      const nodeId = element.dataset.meshNodeCore

      if (nodeId) {
        nodeCoreElements.set(nodeId, element)
      }
    })

    const linkElements = new Map<string, SVGLineElement>()
    svgPanLayerElement.querySelectorAll<SVGLineElement>('[data-mesh-link-id]').forEach((element) => {
      const linkId = element.dataset.meshLinkId

      if (linkId) {
        linkElements.set(linkId, element)
      }
    })

    const newTransitions = transitioningNodes.filter(({ node, phase }) => {
      const key = lifecycleTransitionKey(node.id, phase)

      return !nodeLifecycleAnimationKeysRef.current.has(key)
    })

    if (newTransitions.length === 0) {
      return undefined
    }

    const timelineId = meshLifecycleTimelineIdRef.current
    meshLifecycleTimelineIdRef.current += 1
    const timelineKeys = new Set(newTransitions.map(({ node, phase }) => lifecycleTransitionKey(node.id, phase)))
    const transitionIndexByNodeId = new Map(newTransitions.map(({ node }, index) => [node.id, index]))
    const animatedLinkIds = new Set<string>()
    const enteringCoreRestingShadows = new Map<HTMLElement, string>()
    const pulsedCoreElements = new Set<HTMLElement>()
    const enteringLinkElements = new Set<SVGLineElement>()

    for (const key of timelineKeys) {
      nodeLifecycleAnimationKeysRef.current.add(key)
    }

    const timeline = createTimeline({
      defaults: { ease: 'outQuart' },
      onComplete: () => {
        for (const [element, boxShadow] of enteringCoreRestingShadows) {
          element.style.removeProperty('opacity')
          element.style.removeProperty('scale')
          element.style.boxShadow = boxShadow
        }

        for (const element of pulsedCoreElements) {
          if (!enteringCoreRestingShadows.has(element)) {
            element.style.removeProperty('opacity')
            element.style.removeProperty('scale')
          }
        }

        for (const element of enteringLinkElements) {
          element.style.removeProperty('opacity')
          element.style.removeProperty('stroke-dashoffset')
        }

        meshLifecycleTimelineRecordsRef.current.delete(timelineId)
      }
    })

    newTransitions.forEach(({ node, phase }, index) => {
      const nodeCoreElement = nodeCoreElements.get(node.id)

      if (!nodeCoreElement) {
        return
      }

      const connectedLinks = links.filter((link) => link.source.id === node.id || link.target.id === node.id)
      const connectedLinkElements = connectedLinks
        .filter((link) => {
          if (animatedLinkIds.has(link.id)) {
            return false
          }

          const otherNodeId = link.source.id === node.id ? link.target.id : link.source.id
          const otherTransitionIndex = transitionIndexByNodeId.get(otherNodeId)

          if (otherTransitionIndex === undefined) {
            return true
          }

          const otherPhase = nodeLifecyclePhase(otherNodeId)

          if (otherPhase !== phase) {
            return phase === 'leaving'
          }

          return phase === 'entering' ? index >= otherTransitionIndex : index <= otherTransitionIndex
        })
        .map((link) => {
          const element = linkElements.get(link.id)

          if (element) {
            animatedLinkIds.add(link.id)
          }

          return element
        })
        .filter(isDefined)

      if (phase === 'entering') {
        const start = index * NODE_JOIN_STAGGER_MS
        const nodeColor = nodeCoreElement.style.color || 'currentColor'

        enteringCoreRestingShadows.set(nodeCoreElement, nodeCoreElement.style.boxShadow)

        timeline.set(
          nodeCoreElement,
          {
            opacity: 0,
            scale: 0.54,
            boxShadow: `0 0 0 0 color-mix(in oklab, ${nodeColor} 0%, transparent)`
          },
          start
        )
        timeline.add(
          nodeCoreElement,
          {
            opacity: [0, 1, 0.98],
            scale: [0.6, 1.34, 1.08],
            boxShadow: [
              `0 0 0 0 color-mix(in oklab, ${nodeColor} 0%, transparent)`,
              `0 0 30px 3px color-mix(in oklab, ${nodeColor} 34%, transparent)`,
              `0 0 14px 1px color-mix(in oklab, ${nodeColor} 20%, transparent)`
            ],
            duration: NODE_JOIN_DURATION_MS
          },
          start
        )

        if (connectedLinkElements.length > 0) {
          for (const linkElement of connectedLinkElements) {
            enteringLinkElements.add(linkElement)
          }

          timeline.set(connectedLinkElements, { opacity: 0, strokeDashoffset: 1 }, start)
          timeline.add(
            connectedLinkElements,
            {
              opacity: [0, 0.62],
              strokeDashoffset: [1, 0],
              duration: LINK_JOIN_DURATION_MS
            },
            start + NODE_JOIN_DURATION_MS
          )
        }

        const connectedCoreElements = connectedLinks
          .map((link) => (link.source.id === node.id ? link.target.id : link.source.id))
          .filter((nodeId) => nodeLifecyclePhase(nodeId) === 'present')
          .map((nodeId) => nodeCoreElements.get(nodeId))
          .filter(isDefined)

        if (connectedCoreElements.length > 0) {
          for (const element of connectedCoreElements) {
            pulsedCoreElements.add(element)
          }

          timeline.add(
            connectedCoreElements,
            {
              opacity: [0.98, 1, 0.98],
              scale: [1, 1.12, 1],
              duration: CONNECTED_NODE_PULSE_DURATION_MS
            },
            start + CONNECTED_NODE_PULSE_DELAY_MS
          )
        }

        return
      }

      const start = index * NODE_LEAVE_STAGGER_MS
      const nodeColor = nodeCoreElement.style.color || 'currentColor'

      if (connectedLinkElements.length > 0) {
        for (const linkElement of connectedLinkElements) {
          timeline.add(
            linkElement,
            {
              opacity: [currentElementOpacity(linkElement, 0.62), 0],
              strokeDashoffset: [currentStrokeDashOffset(linkElement, 0), -1],
              duration: LINK_LEAVE_DURATION_MS,
              ease: 'inQuart'
            },
            start
          )
        }
      }

      timeline.add(
        nodeCoreElement,
        {
          opacity: [currentElementOpacity(nodeCoreElement, 0.98), 0],
          scale: [currentElementScale(nodeCoreElement, 1.08), 0.72],
          boxShadow: [
            nodeCoreElement.style.boxShadow || `0 0 14px 1px color-mix(in oklab, ${nodeColor} 20%, transparent)`,
            `0 0 0 0 color-mix(in oklab, ${nodeColor} 0%, transparent)`
          ],
          duration: NODE_LEAVE_DURATION_MS,
          ease: 'inQuart'
        },
        start + NODE_LEAVE_DELAY_MS
      )
    })

    meshLifecycleTimelineRecordsRef.current.set(timelineId, { keys: timelineKeys, timeline })

    return undefined
  }, [
    links,
    meshLifecycleTimelineIdRef,
    meshLifecycleTimelineRecordsRef,
    nodeLayerRef,
    nodeLifecycleAnimationKeysRef,
    nodeLifecyclePhase,
    reduceMotion,
    renderNodes,
    svgPanLayerRef
  ])

  useLayoutEffect(() => {
    const currentLinkIds = new Set(links.map((link) => link.id))
    const currentLinksById = new Map(links.map((link) => [link.id, link]))
    const previousLinkIds = previousLinkIdsRef.current

    previousLinkIdsRef.current = currentLinkIds

    for (const linkId of [...linkRestoreAnimationIdsRef.current]) {
      if (!currentLinkIds.has(linkId)) {
        linkRestoreAnimationIdsRef.current.delete(linkId)
      }
    }

    for (const [recordId, record] of linkRestoreTimelineRecordsRef.current) {
      const stillRestoring = [...record.linkIds].every((linkId) => {
        const link = currentLinksById.get(linkId)

        return link && linkLifecyclePhase(link.source.id, link.target.id) === 'present'
      })

      if (!stillRestoring) {
        record.timeline.revert()

        for (const linkId of record.linkIds) {
          linkRestoreAnimationIdsRef.current.delete(linkId)
        }

        linkRestoreTimelineRecordsRef.current.delete(recordId)
      }
    }

    if (reduceMotion) {
      for (const record of linkRestoreTimelineRecordsRef.current.values()) {
        record.timeline.revert()
      }

      linkRestoreTimelineRecordsRef.current.clear()
      linkRestoreAnimationIdsRef.current.clear()
      return undefined
    }

    if (previousLinkIds.size === 0) {
      return undefined
    }

    const restoredLinks = links.filter(
      (link) =>
        !previousLinkIds.has(link.id) &&
        !linkRestoreAnimationIdsRef.current.has(link.id) &&
        linkLifecyclePhase(link.source.id, link.target.id) === 'present'
    )

    if (restoredLinks.length === 0) {
      return undefined
    }

    const svgPanLayerElement = svgPanLayerRef.current

    if (!svgPanLayerElement) {
      return undefined
    }

    const linkElements = new Map<string, SVGLineElement>()
    svgPanLayerElement.querySelectorAll<SVGLineElement>('[data-mesh-link-id]').forEach((element) => {
      const linkId = element.dataset.meshLinkId

      if (linkId) {
        linkElements.set(linkId, element)
      }
    })
    const restoredLinkElements = restoredLinks.map((link) => linkElements.get(link.id)).filter(isDefined)

    if (restoredLinkElements.length === 0) {
      return undefined
    }

    const timelineId = linkRestoreTimelineIdRef.current
    linkRestoreTimelineIdRef.current += 1

    const restoredLinkIds = new Set(restoredLinks.map((link) => link.id))

    for (const linkId of restoredLinkIds) {
      linkRestoreAnimationIdsRef.current.add(linkId)
    }

    const timeline = createTimeline({
      defaults: { ease: 'outQuart' },
      onComplete: () => {
        for (const element of restoredLinkElements) {
          element.style.removeProperty('opacity')
          element.style.removeProperty('stroke-dashoffset')
        }

        for (const linkId of restoredLinkIds) {
          linkRestoreAnimationIdsRef.current.delete(linkId)
        }

        linkRestoreTimelineRecordsRef.current.delete(timelineId)
      }
    })

    timeline.set(restoredLinkElements, { opacity: 0, strokeDashoffset: 1 }, 0)
    timeline.add(
      restoredLinkElements,
      {
        opacity: [0, 0.62],
        strokeDashoffset: [1, 0],
        duration: LINK_JOIN_DURATION_MS
      },
      0
    )

    linkRestoreTimelineRecordsRef.current.set(timelineId, { linkIds: restoredLinkIds, timeline })

    return undefined
  }, [
    linkRestoreAnimationIdsRef,
    linkRestoreTimelineIdRef,
    linkRestoreTimelineRecordsRef,
    links,
    linkLifecyclePhase,
    previousLinkIdsRef,
    reduceMotion,
    svgPanLayerRef
  ])

  useEffect(
    () => () => {
      for (const timeout of nodeLifecycleTimeoutsRef.current.values()) {
        window.clearTimeout(timeout)
      }

      nodeLifecycleTimeoutsRef.current.clear()
      nodeLifecycleAnimationKeysRef.current.clear()
      previousLinkIdsRef.current.clear()
      linkRestoreAnimationIdsRef.current.clear()

      for (const record of meshLifecycleTimelineRecordsRef.current.values()) {
        record.timeline.revert()
      }

      meshLifecycleTimelineRecordsRef.current.clear()

      for (const record of linkRestoreTimelineRecordsRef.current.values()) {
        record.timeline.revert()
      }

      linkRestoreTimelineRecordsRef.current.clear()
    },
    [
      linkRestoreAnimationIdsRef,
      linkRestoreTimelineRecordsRef,
      meshLifecycleTimelineRecordsRef,
      nodeLifecycleAnimationKeysRef,
      nodeLifecycleTimeoutsRef,
      previousLinkIdsRef
    ]
  )

  useEffect(() => {
    if (openNodeId && !currentRenderNodeIds.has(openNodeId)) {
      setOpenNodeId(undefined)
    }

    if (localHoveredNodeId && !currentRenderNodeIds.has(localHoveredNodeId)) {
      setLocalHoveredNodeId(undefined)
    }
  }, [currentRenderNodeIds, localHoveredNodeId, openNodeId, setLocalHoveredNodeId, setOpenNodeId])
}
