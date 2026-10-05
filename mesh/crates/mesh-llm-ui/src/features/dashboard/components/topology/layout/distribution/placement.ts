import { TAU } from '@/features/dashboard/components/topology/helpers'
import { countBadTopologyEdgeCrossings } from '@/features/dashboard/components/topology/layout/distribution/crossings'
import type {
  TopologyAngularPlacement,
  TopologyGraphEdge,
  TopologyGraphNode
} from '@/features/dashboard/components/topology/layout/distribution/pair-plan'
import { buildTopologyPairPlan } from '@/features/dashboard/components/topology/layout/distribution/pair-plan'

const CROSSING_EPSILON = 1e-6
const CROSSING_SWAP_PASSES = 4

type Point = { x: number; y: number }

type WorkingPlacement<TNode extends { id: string }> = TopologyAngularPlacement<TNode> & {
  originalX: number
  originalY: number
}

type WorkingBand<TNode extends { id: string }> = {
  locked: WorkingPlacement<TNode>[]
  movableOrder: WorkingPlacement<TNode>[]
  slotAngles: number[]
}

function normalizeAngle(angle: number) {
  const wrapped = angle % TAU
  return wrapped < 0 ? wrapped + TAU : wrapped
}

function angleFromPoint(point: Point, center: Point) {
  return normalizeAngle(Math.atan2(point.y - center.y, point.x - center.x))
}

function placementComparator<TNode extends { id: string }>(
  left: TopologyAngularPlacement<TNode>,
  right: TopologyAngularPlacement<TNode>
) {
  return left.angle - right.angle || left.node.id.localeCompare(right.node.id)
}

function buildAdjacency(edges: TopologyGraphEdge[]) {
  const adjacency = new Map<string, Set<string>>()

  for (const edge of edges) {
    if (!adjacency.has(edge.leftId)) adjacency.set(edge.leftId, new Set())
    if (!adjacency.has(edge.rightId)) adjacency.set(edge.rightId, new Set())
    adjacency.get(edge.leftId)?.add(edge.rightId)
    adjacency.get(edge.rightId)?.add(edge.leftId)
  }

  return adjacency
}

function materializeBand<TNode extends { id: string }>(band: WorkingBand<TNode>): TopologyAngularPlacement<TNode>[] {
  const placedMovable = band.movableOrder.map((entry, index) => {
    const angle = band.slotAngles[index] ?? entry.angle
    const point = entry.positionAtAngle(angle)
    return {
      ...entry,
      angle,
      x: point.x,
      y: point.y
    }
  })

  return [...band.locked, ...placedMovable].sort(placementComparator)
}

function materializeBands<TNode extends { id: string }>(
  bands: WorkingBand<TNode>[]
): TopologyAngularPlacement<TNode>[][] {
  return bands.map((band) => materializeBand(band))
}

function flattenBands<TNode extends { id: string }>(bands: TopologyAngularPlacement<TNode>[][]): TopologyGraphNode[] {
  return bands.flat().map((entry) => ({
    id: entry.node.id,
    x: entry.x,
    y: entry.y,
    role: entry.role,
    selectedModelMatch: entry.selectedModelMatch
  }))
}

function scoreLayout<TNode extends { id: string }>(
  pairPlan: TopologyGraphEdge[],
  bands: WorkingBand<TNode>[],
  selfNode: TopologyGraphNode,
  originalPositions: Map<string, Point>
) {
  const materializedBands = materializeBands(bands)
  const graphNodes = [...flattenBands(materializedBands), selfNode]
  const nodeById = new Map(graphNodes.map((node) => [node.id, node]))
  const edgeLength = pairPlan.reduce((total, edge) => {
    const left = nodeById.get(edge.leftId)
    const right = nodeById.get(edge.rightId)
    if (!left || !right) return total
    return total + Math.hypot(left.x - right.x, left.y - right.y)
  }, 0)
  const movement = materializedBands.reduce((total, band) => {
    return (
      total +
      band.reduce((bandTotal, entry) => {
        if (entry.locked) return bandTotal
        const original = originalPositions.get(entry.node.id)
        if (!original) return bandTotal
        return bandTotal + Math.hypot(entry.x - original.x, entry.y - original.y)
      }, 0)
    )
  }, 0)

  return {
    badCrossings: countBadTopologyEdgeCrossings(pairPlan, graphNodes),
    edgeLength,
    movement
  }
}

function compareLayoutScore(left: ReturnType<typeof scoreLayout>, right: ReturnType<typeof scoreLayout>) {
  if (left.badCrossings !== right.badCrossings) {
    return left.badCrossings - right.badCrossings
  }
  if (Math.abs(left.edgeLength - right.edgeLength) > CROSSING_EPSILON) {
    return left.edgeLength - right.edgeLength
  }
  if (Math.abs(left.movement - right.movement) > CROSSING_EPSILON) {
    return left.movement - right.movement
  }
  return 0
}

export function optimizeTopologyPlacementForPlan<TNode extends { id: string }>(
  bands: Array<Array<TopologyAngularPlacement<TNode>>>,
  selfNode: TopologyGraphNode,
  pairPlan: TopologyGraphEdge[]
) {
  if (!pairPlan.length) {
    return bands.map((band) => [...band].sort(placementComparator))
  }

  const originalPositions = new Map<string, Point>()
  const workingBands = bands.map<WorkingBand<TNode>>((band) => {
    const sortedBand = [...band]
      .map((entry) => {
        originalPositions.set(entry.node.id, { x: entry.x, y: entry.y })
        return {
          ...entry,
          originalX: entry.x,
          originalY: entry.y
        } satisfies WorkingPlacement<TNode>
      })
      .sort(placementComparator)

    return {
      locked: sortedBand.filter((entry) => entry.locked),
      movableOrder: sortedBand.filter((entry) => !entry.locked),
      slotAngles: sortedBand.filter((entry) => !entry.locked).map((entry) => entry.angle)
    }
  })

  const currentNodes = [...flattenBands(materializeBands(workingBands)), selfNode]
  const currentNodeById = new Map(currentNodes.map((node) => [node.id, node]))
  const adjacency = buildAdjacency(pairPlan)
  const centerPoint = { x: selfNode.x, y: selfNode.y }

  for (const band of workingBands) {
    if (band.movableOrder.length < 2) continue

    const targetAngles = new Map<string, number>()
    for (const entry of band.movableOrder) {
      const neighbors = [...(adjacency.get(entry.node.id) ?? [])]
        .map((neighborId) => currentNodeById.get(neighborId))
        .filter((neighbor): neighbor is TopologyGraphNode => neighbor != null)
      if (!neighbors.length) {
        targetAngles.set(entry.node.id, entry.angle)
        continue
      }

      let sumX = 0
      let sumY = 0
      for (const neighbor of neighbors) {
        const neighborAngle = angleFromPoint(neighbor, centerPoint)
        const weight = 1 / Math.max(Math.hypot(neighbor.x - centerPoint.x, neighbor.y - centerPoint.y), 0.05)
        sumX += Math.cos(neighborAngle) * weight
        sumY += Math.sin(neighborAngle) * weight
      }
      targetAngles.set(
        entry.node.id,
        Math.abs(sumX) <= CROSSING_EPSILON && Math.abs(sumY) <= CROSSING_EPSILON
          ? entry.angle
          : normalizeAngle(Math.atan2(sumY, sumX))
      )
    }

    band.movableOrder.sort(
      (left, right) =>
        (targetAngles.get(left.node.id) ?? left.angle) - (targetAngles.get(right.node.id) ?? right.angle) ||
        left.node.id.localeCompare(right.node.id)
    )
  }

  let bestScore = scoreLayout(pairPlan, workingBands, selfNode, originalPositions)

  for (let pass = 0; pass < CROSSING_SWAP_PASSES; pass += 1) {
    let improved = false

    for (let bandIndex = 0; bandIndex < workingBands.length; bandIndex += 1) {
      if (workingBands[bandIndex].movableOrder.length < 2) continue

      for (let swapIndex = 0; swapIndex < workingBands[bandIndex].movableOrder.length - 1; swapIndex += 1) {
        const band = workingBands[bandIndex]
        const nextOrder = [...band.movableOrder]
        ;[nextOrder[swapIndex], nextOrder[swapIndex + 1]] = [nextOrder[swapIndex + 1], nextOrder[swapIndex]]

        const candidateBands = workingBands.map((entry, index) =>
          index === bandIndex ? { ...entry, movableOrder: nextOrder } : entry
        )
        const candidateScore = scoreLayout(pairPlan, candidateBands, selfNode, originalPositions)
        if (compareLayoutScore(candidateScore, bestScore) < 0) {
          workingBands[bandIndex] = { ...band, movableOrder: nextOrder }
          bestScore = candidateScore
          improved = true
        }
      }
    }

    if (!improved) {
      break
    }
  }

  return materializeBands(workingBands)
}

export function reduceTopologyCrossings<TNode extends { id: string }>(
  bands: Array<Array<TopologyAngularPlacement<TNode>>>,
  selfNode: TopologyGraphNode,
  centerNodeId?: string
) {
  const pairPlan = buildTopologyPairPlan([...flattenBands(bands), selfNode], centerNodeId)
  return optimizeTopologyPlacementForPlan(bands, selfNode, pairPlan)
}
