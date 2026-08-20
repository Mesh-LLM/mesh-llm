import type {
  TopologyGraphEdge,
  TopologyGraphNode
} from '@/features/dashboard/components/topology/layout/distribution/pair-plan'

const CROSSING_EPSILON = 1e-6
type Point = { x: number; y: number }

function orientation(start: Point, middle: Point, end: Point) {
  const value = (middle.y - start.y) * (end.x - middle.x) - (middle.x - start.x) * (end.y - middle.y)
  if (Math.abs(value) <= CROSSING_EPSILON) return 0
  return value > 0 ? 1 : -1
}

function isInteriorPointOnSegment(point: Point, start: Point, end: Point) {
  if (
    (Math.abs(point.x - start.x) <= CROSSING_EPSILON && Math.abs(point.y - start.y) <= CROSSING_EPSILON) ||
    (Math.abs(point.x - end.x) <= CROSSING_EPSILON && Math.abs(point.y - end.y) <= CROSSING_EPSILON)
  ) {
    return false
  }

  return (
    point.x <= Math.max(start.x, end.x) + CROSSING_EPSILON &&
    point.x >= Math.min(start.x, end.x) - CROSSING_EPSILON &&
    point.y <= Math.max(start.y, end.y) + CROSSING_EPSILON &&
    point.y >= Math.min(start.y, end.y) - CROSSING_EPSILON
  )
}

function segmentsCrossExcludingEndpoints(leftStart: Point, leftEnd: Point, rightStart: Point, rightEnd: Point) {
  const leftRightStart = orientation(leftStart, leftEnd, rightStart)
  const leftRightEnd = orientation(leftStart, leftEnd, rightEnd)
  const rightLeftStart = orientation(rightStart, rightEnd, leftStart)
  const rightLeftEnd = orientation(rightStart, rightEnd, leftEnd)

  if (leftRightStart !== leftRightEnd && rightLeftStart !== rightLeftEnd) {
    return true
  }

  if (leftRightStart === 0 && isInteriorPointOnSegment(rightStart, leftStart, leftEnd)) {
    return true
  }
  if (leftRightEnd === 0 && isInteriorPointOnSegment(rightEnd, leftStart, leftEnd)) {
    return true
  }
  if (rightLeftStart === 0 && isInteriorPointOnSegment(leftStart, rightStart, rightEnd)) {
    return true
  }
  if (rightLeftEnd === 0 && isInteriorPointOnSegment(leftEnd, rightStart, rightEnd)) {
    return true
  }

  return false
}

export function countBadTopologyEdgeCrossings(edges: TopologyGraphEdge[], nodes: TopologyGraphNode[]) {
  const nodeById = new Map(nodes.map((node) => [node.id, node]))
  let crossings = 0

  for (let leftIndex = 0; leftIndex < edges.length; leftIndex += 1) {
    const leftEdge = edges[leftIndex]
    const leftStart = nodeById.get(leftEdge.leftId)
    const leftEnd = nodeById.get(leftEdge.rightId)
    if (!leftStart || !leftEnd) continue

    for (let rightIndex = leftIndex + 1; rightIndex < edges.length; rightIndex += 1) {
      const rightEdge = edges[rightIndex]
      if (
        leftEdge.leftId === rightEdge.leftId ||
        leftEdge.leftId === rightEdge.rightId ||
        leftEdge.rightId === rightEdge.leftId ||
        leftEdge.rightId === rightEdge.rightId
      ) {
        continue
      }

      const rightStart = nodeById.get(rightEdge.leftId)
      const rightEnd = nodeById.get(rightEdge.rightId)
      if (!rightStart || !rightEnd) continue

      if (segmentsCrossExcludingEndpoints(leftStart, leftEnd, rightStart, rightEnd)) {
        crossings += 1
      }
    }
  }

  return crossings
}
