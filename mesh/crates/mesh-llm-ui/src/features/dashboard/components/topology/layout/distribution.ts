/**
 * Compatibility facade for topology distribution policies.
 *
 * Pair planning, angular crossing reduction, perimeter/latency placement, and
 * overlap resolution live in ownership-focused modules while this path remains
 * stable for existing dashboard and test imports.
 */
export type { TopologyAngularPlacement, TopologyGraphEdge, TopologyGraphNode } from './distribution/pair-plan'
export { buildTopologyPairPlan } from './distribution/pair-plan'
export { countBadTopologyEdgeCrossings } from './distribution/crossings'
export { optimizeTopologyPlacementForPlan, reduceTopologyCrossings } from './distribution/placement'
export { distributeLatencyBand, distributePerimeterClients, nodeSize } from './distribution/perimeter'
export { resolveNodeOverlap } from './distribution/overlap'
