type Point = {
  x: number
  y: number
}

/** Binary min-heap keyed by distance for O((V+E) log V) Dijkstra. */
class MinHeap {
  private heap: [number, string][] = []

  push(distance: number, id: string) {
    this.heap.push([distance, id])
    this.bubbleUp(this.heap.length - 1)
  }

  pop(): [number, string] | undefined {
    if (this.heap.length === 0) return undefined
    const min = this.heap[0]
    const last = this.heap.pop()!
    if (this.heap.length > 0) {
      this.heap[0] = last
      this.sinkDown(0)
    }
    return min
  }

  get size() {
    return this.heap.length
  }

  private bubbleUp(i: number) {
    while (i > 0) {
      const parent = (i - 1) >> 1
      if (this.heap[parent][0] <= this.heap[i][0]) break
      ;[this.heap[parent], this.heap[i]] = [this.heap[i], this.heap[parent]]
      i = parent
    }
  }

  private sinkDown(i: number) {
    const n = this.heap.length
    while (true) {
      let smallest = i
      const left = 2 * i + 1
      const right = 2 * i + 2
      if (left < n && this.heap[left][0] < this.heap[smallest][0]) smallest = left
      if (right < n && this.heap[right][0] < this.heap[smallest][0]) smallest = right
      if (smallest === i) break
      ;[this.heap[smallest], this.heap[i]] = [this.heap[i], this.heap[smallest]]
      i = smallest
    }
  }
}

export type TopologyGraphNode = {
  id: string
  x: number
  y: number
  role: string
  selectedModelMatch: boolean
}

export type TopologyGraphEdge = {
  pairKey: string
  leftId: string
  rightId: string
  distance: number
  threshold: number
}

export type TopologyAngularPlacement<TNode extends { id: string }> = {
  node: TNode
  x: number
  y: number
  angle: number
  role: string
  selectedModelMatch: boolean
  locked?: boolean
  latencyNorm?: number
  positionAtAngle: (angle: number) => Point
}

function isClientRole(role: string) {
  return role === 'Client'
}

function pairKeyFor(leftId: string, rightId: string) {
  return [leftId, rightId].sort((first, second) => first.localeCompare(second)).join('::')
}

function proximityThreshold(
  left: Pick<TopologyGraphNode, 'id' | 'role'>,
  right: Pick<TopologyGraphNode, 'id' | 'role'>,
  centerNodeId?: string
) {
  let threshold = 0.21
  if (left.id === centerNodeId || right.id === centerNodeId) threshold += 0.03
  if (isClientRole(left.role) || isClientRole(right.role)) threshold += 0.08
  if (left.role === 'Host' || right.role === 'Host') threshold += 0.03
  return threshold
}

function clientAnchorThreshold(
  client: Pick<TopologyGraphNode, 'id' | 'role'>,
  anchor: Pick<TopologyGraphNode, 'id' | 'role'>,
  centerNodeId?: string
) {
  const baseThreshold = proximityThreshold(client, anchor, centerNodeId)
  let threshold = Math.max(baseThreshold + 0.06, 0.35)
  if (anchor.role === 'Host') threshold = Math.max(threshold, 0.39)
  else if (anchor.role === 'Serving') threshold = Math.max(threshold, 0.37)
  else if (anchor.role === 'Worker') threshold = Math.max(threshold, 0.36)
  if (anchor.id === centerNodeId) threshold += 0.01
  return threshold
}

// Ownership/invariants: this module owns topology pair planning only. Angular placement and
// crossing-swap reduction live in `placement.ts`, edge-crossing detection in `crossings.ts`,
// perimeter client placement in `perimeter.ts`, and node-overlap resolution/sizing in
// `overlap.ts`. Render-space edge geometry belongs in `render/line-builders.ts`.

export function buildTopologyPairPlan(nodes: TopologyGraphNode[], centerNodeId?: string): TopologyGraphEdge[] {
  if (nodes.length < 2) {
    return []
  }

  const backboneNodes = nodes.filter((node) => !isClientRole(node.role))
  if (!backboneNodes.length) {
    return []
  }

  const rootNode =
    (centerNodeId ? backboneNodes.find((node) => node.id === centerNodeId) : undefined) ?? backboneNodes[0]

  const rootRadius = (node: TopologyGraphNode) => Math.hypot(node.x - rootNode.x, node.y - rootNode.y)
  const rootEdgePenalty = (left: TopologyGraphNode, right: TopologyGraphNode) => {
    if (left.id !== rootNode.id && right.id !== rootNode.id) return 0
    const fartherRadius = Math.max(rootRadius(left), rootRadius(right))
    return Math.max(0, fartherRadius - 0.16) * 0.8
  }
  const backboneThreshold = (left: TopologyGraphNode, right: TopologyGraphNode) => {
    let threshold = Math.max(proximityThreshold(left, right, centerNodeId) + 0.04, 0.3)
    if (left.id === rootNode.id || right.id === rootNode.id) threshold += 0.02
    return threshold
  }

  type CandidateEdge = TopologyGraphEdge & {
    cost: number
  }

  const baseEdgesByKey = new Map<string, CandidateEdge>()
  const backboneById = new Map(backboneNodes.map((node) => [node.id, node]))
  const upsertBackboneEdge = (edge: CandidateEdge) => {
    const existing = baseEdgesByKey.get(edge.pairKey)
    if (!existing || edge.cost < existing.cost || (edge.cost === existing.cost && edge.distance < existing.distance)) {
      baseEdgesByKey.set(edge.pairKey, edge)
    }
  }

  for (const node of backboneNodes) {
    const neighborCandidates = backboneNodes
      .filter((other) => other.id !== node.id)
      .map((other) => {
        const distance = Math.hypot(node.x - other.x, node.y - other.y)
        const threshold = backboneThreshold(node, other)
        return {
          other,
          distance,
          threshold,
          radialDelta: rootRadius(other) - rootRadius(node)
        }
      })
      .sort((left, right) => left.distance - right.distance || left.other.id.localeCompare(right.other.id))

    if (!neighborCandidates.length) continue

    const selectedCandidates = new Map<string, (typeof neighborCandidates)[number]>()
    const nearestOverall = neighborCandidates[0]
    const nearestInward = neighborCandidates.find((candidate) => candidate.radialDelta < -0.01)
    const inThreshold = neighborCandidates.filter((candidate) => candidate.distance <= candidate.threshold)
    const keepCount = node.id === rootNode.id ? 4 : 3
    const isOuterBackboneNode = rootRadius(node) > 0.22
    const bestInwardRelay = inThreshold.find(
      (candidate) => candidate.other.id !== rootNode.id && candidate.radialDelta < -0.015
    )
    const suppressRootShortcut = node.id !== rootNode.id && isOuterBackboneNode && bestInwardRelay != null

    for (const candidate of inThreshold.slice(0, keepCount)) {
      if (suppressRootShortcut && candidate.other.id === rootNode.id) continue
      selectedCandidates.set(candidate.other.id, candidate)
    }
    if (nearestOverall && !(suppressRootShortcut && nearestOverall.other.id === rootNode.id)) {
      selectedCandidates.set(nearestOverall.other.id, nearestOverall)
    }
    if (nearestInward) {
      selectedCandidates.set(nearestInward.other.id, nearestInward)
    }

    for (const candidate of selectedCandidates.values()) {
      upsertBackboneEdge({
        pairKey: pairKeyFor(node.id, candidate.other.id),
        leftId: node.id,
        rightId: candidate.other.id,
        distance: candidate.distance,
        threshold: candidate.threshold,
        cost: candidate.distance + rootEdgePenalty(node, candidate.other)
      })
    }
  }

  const runShortestPathTree = (bridgeEdges: CandidateEdge[]) => {
    const adjacency = new Map<string, CandidateEdge[]>()
    for (const node of backboneNodes) {
      adjacency.set(node.id, [])
    }

    for (const edge of [...baseEdgesByKey.values(), ...bridgeEdges]) {
      adjacency.get(edge.leftId)?.push(edge)
      adjacency.get(edge.rightId)?.push(edge)
    }

    const distances = new Map(backboneNodes.map((node) => [node.id, Number.POSITIVE_INFINITY]))
    const previous = new Map(backboneNodes.map((node) => [node.id, null as string | null]))
    const visited = new Set<string>()
    distances.set(rootNode.id, 0)

    const heap = new MinHeap()
    heap.push(0, rootNode.id)

    while (heap.size > 0) {
      const [currentDistance, currentId] = heap.pop()!
      if (visited.has(currentId)) continue
      if (!Number.isFinite(currentDistance)) break
      visited.add(currentId)

      for (const edge of adjacency.get(currentId) ?? []) {
        const neighborId = edge.leftId === currentId ? edge.rightId : edge.leftId
        if (visited.has(neighborId)) continue

        const nextDistance = currentDistance + edge.cost
        const knownDistance = distances.get(neighborId) ?? Number.POSITIVE_INFINITY
        if (nextDistance < knownDistance) {
          distances.set(neighborId, nextDistance)
          previous.set(neighborId, currentId)
          heap.push(nextDistance, neighborId)
        }
      }
    }

    return { distances, previous }
  }

  const bridgeEdges: CandidateEdge[] = []
  let backboneTree = runShortestPathTree(bridgeEdges)
  while (
    backboneNodes.some((node) => !Number.isFinite(backboneTree.distances.get(node.id) ?? Number.POSITIVE_INFINITY))
  ) {
    const reachable = backboneNodes.filter((node) =>
      Number.isFinite(backboneTree.distances.get(node.id) ?? Number.POSITIVE_INFINITY)
    )
    const unreachable = backboneNodes.filter(
      (node) => !Number.isFinite(backboneTree.distances.get(node.id) ?? Number.POSITIVE_INFINITY)
    )

    let bestBridge: CandidateEdge | null = null
    for (const left of reachable) {
      for (const right of unreachable) {
        const pairKey = pairKeyFor(left.id, right.id)
        if (baseEdgesByKey.has(pairKey) || bridgeEdges.some((edge) => edge.pairKey === pairKey)) {
          continue
        }

        const distance = Math.hypot(left.x - right.x, left.y - right.y)
        const bridge = {
          pairKey,
          leftId: left.id,
          rightId: right.id,
          distance,
          threshold: backboneThreshold(left, right) * 1.2,
          cost: distance + rootEdgePenalty(left, right)
        }

        if (
          !bestBridge ||
          bridge.cost < bestBridge.cost ||
          (bridge.cost === bestBridge.cost && bridge.pairKey.localeCompare(bestBridge.pairKey) < 0)
        ) {
          bestBridge = bridge
        }
      }
    }

    if (!bestBridge) break
    bridgeEdges.push(bestBridge)
    backboneTree = runShortestPathTree(bridgeEdges)
  }

  const allBackboneEdges = new Map(baseEdgesByKey)
  for (const edge of bridgeEdges) {
    allBackboneEdges.set(edge.pairKey, edge)
  }

  const plannedPairs: TopologyGraphEdge[] = []
  const renderedPairs = new Set<string>()
  const pushPlannedPair = (edge: TopologyGraphEdge) => {
    if (renderedPairs.has(edge.pairKey)) return
    renderedPairs.add(edge.pairKey)
    plannedPairs.push(edge)
  }

  for (const node of backboneNodes) {
    if (node.id === rootNode.id) continue
    const previousId = backboneTree.previous.get(node.id)
    if (!previousId) continue

    const pairKey = pairKeyFor(node.id, previousId)
    const edge = allBackboneEdges.get(pairKey)
    const previousNode = backboneById.get(previousId)
    if (!edge || !previousNode) continue

    pushPlannedPair({
      pairKey,
      leftId: previousNode.id,
      rightId: node.id,
      distance: edge.distance,
      threshold: edge.threshold
    })
  }

  type ClientAttachment = {
    pairKey: string
    anchor: TopologyGraphNode
    distance: number
    threshold: number
    score: number
  }

  const isBetterClientAttachment = (nextCandidate: ClientAttachment, currentCandidate: ClientAttachment | null) =>
    !currentCandidate ||
    nextCandidate.score < currentCandidate.score ||
    (nextCandidate.score === currentCandidate.score &&
      nextCandidate.pairKey.localeCompare(currentCandidate.pairKey) < 0)

  for (const client of nodes) {
    if (!isClientRole(client.role)) continue

    let bestCandidate: ClientAttachment | null = null
    let bestEmergencyCandidate: ClientAttachment | null = null

    for (const anchor of backboneNodes) {
      if (anchor.id === client.id) continue

      const pairKey = pairKeyFor(client.id, anchor.id)
      if (renderedPairs.has(pairKey)) continue

      const distance = Math.hypot(client.x - anchor.x, client.y - anchor.y)
      const threshold = clientAnchorThreshold(client, anchor, centerNodeId)
      const score =
        distance +
        (anchor.id === rootNode.id ? 0.02 : 0) -
        (anchor.role === 'Host' ? 0.045 : anchor.role === 'Serving' ? 0.03 : anchor.role === 'Worker' ? 0.026 : 0) -
        (anchor.selectedModelMatch ? 0.008 : 0)

      const candidate = {
        pairKey,
        anchor,
        distance,
        threshold,
        score
      }

      if (isBetterClientAttachment(candidate, bestEmergencyCandidate)) {
        bestEmergencyCandidate = candidate
      }
      if (distance <= threshold && isBetterClientAttachment(candidate, bestCandidate)) {
        bestCandidate = candidate
      }
    }

    const attachment = bestCandidate ?? bestEmergencyCandidate
    if (!attachment) continue

    pushPlannedPair({
      pairKey: attachment.pairKey,
      leftId: attachment.anchor.id,
      rightId: client.id,
      distance: attachment.distance,
      threshold: attachment.threshold * (bestCandidate ? 1 : 1.12)
    })
  }

  return plannedPairs
}
