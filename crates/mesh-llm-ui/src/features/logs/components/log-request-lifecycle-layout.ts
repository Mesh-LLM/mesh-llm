import type { CSSProperties } from 'react'

const NODE_MIN_WIDTH_PX = 116
const CONNECTOR_ICON_RADIUS_PX = 16
const CONNECTOR_GAP_PX = 7.5
const CONNECTOR_OFFSET_PX = CONNECTOR_ICON_RADIUS_PX + CONNECTOR_GAP_PX
const MOBILE_TRACK_WIDTH_PX = 640
const MOBILE_MAX_NODES_PER_PAGE = 3
const DESKTOP_MAX_NODES_PER_PAGE = 6

export function nodesPerPage(trackWidth: number): number {
  if (trackWidth <= 0) return DESKTOP_MAX_NODES_PER_PAGE
  const ceiling = trackWidth < MOBILE_TRACK_WIDTH_PX ? MOBILE_MAX_NODES_PER_PAGE : DESKTOP_MAX_NODES_PER_PAGE
  return Math.max(1, Math.min(ceiling, Math.floor(trackWidth / NODE_MIN_WIDTH_PX)))
}

export function connectorPositionStyle(): CSSProperties {
  return {
    insetInlineStart: `calc(50% + ${CONNECTOR_OFFSET_PX}px)`,
    insetInlineEnd: `calc(-50% + ${CONNECTOR_OFFSET_PX}px)`
  }
}

export function incomingConnectorPositionStyle(): CSSProperties {
  return {
    insetInlineStart: 0,
    insetInlineEnd: `calc(50% + ${CONNECTOR_OFFSET_PX}px)`
  }
}

export function outgoingConnectorPositionStyle(): CSSProperties {
  return {
    insetInlineStart: `calc(50% + ${CONNECTOR_OFFSET_PX}px)`,
    insetInlineEnd: 0
  }
}

export function sparseListStyle(nodeCount: number): CSSProperties | undefined {
  if (nodeCount >= DESKTOP_MAX_NODES_PER_PAGE) return undefined
  return { maxWidth: nodeCount * NODE_MIN_WIDTH_PX }
}
