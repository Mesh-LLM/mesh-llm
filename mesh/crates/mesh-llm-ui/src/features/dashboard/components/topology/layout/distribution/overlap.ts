import { clamp, hashString, TAU } from '@/features/dashboard/components/topology/helpers'
import type { RenderNode } from '@/features/dashboard/components/topology/types'

export function resolveNodeOverlap(nodes: RenderNode[], fixedNodes: RenderNode[] = []) {
  if (nodes.length === 0) return []

  const MAX_ITERATIONS = 50
  const MAX_NODE_SIZE = 38
  const FRICTION_INITIAL = 0.5
  const COOLING_RATE = 0.97
  const CONVERGENCE_THRESHOLD = 0.0005
  const MIN_SEP_BASE = 0.025
  const MIN_SEP_SIZE_FACTOR = 0.003
  const RESTORE_STRENGTH = 0.08
  const GRAVITY_BASE = 0.06
  const GRAVITY_CLIENT = 0.005

  type WorkingNode = {
    id: string
    x: number
    y: number
    size: number
    origX: number
    origY: number
    fixed: boolean
    gravity: number
  }

  const gravityForNode = (node: RenderNode) =>
    node.role === 'Client' ? GRAVITY_CLIENT : GRAVITY_BASE * (node.size / MAX_NODE_SIZE)

  const workingNodes: WorkingNode[] = [
    ...fixedNodes.map((node) => ({
      id: node.id,
      x: node.x,
      y: node.y,
      size: node.size,
      origX: node.x,
      origY: node.y,
      fixed: true,
      gravity: 0
    })),
    ...nodes.map((node) => ({
      id: node.id,
      x: node.x,
      y: node.y,
      size: node.size,
      origX: node.x,
      origY: node.y,
      fixed: false,
      gravity: gravityForNode(node)
    }))
  ]

  const gravityCenterX = fixedNodes.length ? fixedNodes.reduce((sum, n) => sum + n.x, 0) / fixedNodes.length : 0.5
  const gravityCenterY = fixedNodes.length ? fixedNodes.reduce((sum, n) => sum + n.y, 0) / fixedNodes.length : 0.52

  let friction = FRICTION_INITIAL

  for (let iter = 0; iter < MAX_ITERATIONS; iter += 1) {
    let maxDisplacement = 0

    // --- pairwise repulsion for overlapping nodes ---
    for (let i = 0; i < workingNodes.length; i += 1) {
      for (let j = i + 1; j < workingNodes.length; j += 1) {
        const a = workingNodes[i]
        const b = workingNodes[j]
        if (a.fixed && b.fixed) continue

        const dx = b.x - a.x
        const dy = b.y - a.y
        let d = Math.hypot(dx, dy)
        const minSep = MIN_SEP_BASE + (a.size + b.size) * MIN_SEP_SIZE_FACTOR
        if (d >= minSep) continue

        // push direction: from a toward b
        let nx: number
        let ny: number
        if (d < 1e-6) {
          // deterministic jitter for exactly-overlapping nodes
          const jitterAngle = hashString(`${a.id}:${b.id}:jitter`) * TAU
          nx = Math.cos(jitterAngle)
          ny = Math.sin(jitterAngle)
          d = 1e-6
        } else {
          nx = dx / d
          ny = dy / d
        }

        const overlap = minSep - d
        const push = overlap * 0.5 * friction

        if (a.fixed) {
          b.x += nx * push * 2
          b.y += ny * push * 2
          maxDisplacement = Math.max(maxDisplacement, push * 2)
        } else if (b.fixed) {
          a.x -= nx * push * 2
          a.y -= ny * push * 2
          maxDisplacement = Math.max(maxDisplacement, push * 2)
        } else {
          a.x -= nx * push
          a.y -= ny * push
          b.x += nx * push
          b.y += ny * push
          maxDisplacement = Math.max(maxDisplacement, push)
        }
      }
    }

    // --- restoration force toward original band position ---
    for (const node of workingNodes) {
      if (node.fixed) continue
      node.x += (node.origX - node.x) * RESTORE_STRENGTH * friction
      node.y += (node.origY - node.y) * RESTORE_STRENGTH * friction
    }

    // --- gravity toward center, weighted by VRAM (size) ---
    for (const node of workingNodes) {
      if (node.fixed) continue
      node.x += (gravityCenterX - node.x) * node.gravity * friction
      node.y += (gravityCenterY - node.y) * node.gravity * friction
    }

    // --- clamp to layout bounds ---
    for (const node of workingNodes) {
      if (node.fixed) continue
      node.x = clamp(node.x, 0.08, 0.92)
      node.y = clamp(node.y, 0.1, 0.9)
    }

    friction *= COOLING_RATE

    if (maxDisplacement < CONVERGENCE_THRESHOLD) break
  }

  // map resolved positions back onto input nodes
  const resolvedPositions = new Map<string, { x: number; y: number }>()
  for (const wn of workingNodes) {
    if (!wn.fixed) {
      resolvedPositions.set(wn.id, { x: wn.x, y: wn.y })
    }
  }

  return nodes.map((node) => {
    const pos = resolvedPositions.get(node.id)
    return pos ? { ...node, x: pos.x, y: pos.y } : node
  })
}
