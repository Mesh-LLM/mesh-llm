import { clamp, hashString, TAU } from '@/features/dashboard/components/topology/helpers'
import type { TopologyNode } from '@/features/app-shell/lib/topology-types'

export function distributePerimeterClients(nodes: TopologyNode[]) {
  const placed: Array<{ x: number; y: number }> = []

  return nodes.map((node) => {
    const angle = hashString(`${node.id}:angle`) * TAU
    const edgeBias = 0.74 + hashString(`${node.id}:radius`) * 0.16
    const tangentJitter = (hashString(`${node.id}:tangent`) - 0.5) * 0.06
    const radialJitter = (hashString(`${node.id}:radial`) - 0.5) * 0.025
    const positionAtAngle = (nextAngle: number) => {
      const tangentAngle = nextAngle + Math.PI / 2
      const ellipseX = 0.5 + Math.cos(nextAngle) * 0.4 * edgeBias
      const ellipseY = 0.5 + Math.sin(nextAngle) * 0.35 * edgeBias
      return {
        x: clamp(ellipseX + Math.cos(tangentAngle) * tangentJitter + Math.cos(nextAngle) * radialJitter, 0.08, 0.92),
        y: clamp(ellipseY + Math.sin(tangentAngle) * tangentJitter + Math.sin(nextAngle) * radialJitter, 0.1, 0.9)
      }
    }
    let { x, y } = positionAtAngle(angle)

    for (const prior of placed) {
      const dx = x - prior.x
      const dy = y - prior.y
      const distance = Math.hypot(dx, dy)
      if (distance > 0 && distance < 0.032) {
        const push = (0.032 - distance) * 0.5
        x = clamp(x + (dx / distance) * push, 0.08, 0.92)
        y = clamp(y + (dy / distance) * push, 0.1, 0.9)
      }
    }
    placed.push({ x, y })

    return {
      node,
      x,
      y,
      angle,
      positionAtAngle
    }
  })
}

function normalizedLatency(node: TopologyNode, minLatency: number, maxLatency: number) {
  if (node.latencyMs == null || !Number.isFinite(node.latencyMs)) {
    return 0.45
  }
  if (maxLatency <= minLatency) {
    return 0.2
  }
  return clamp((node.latencyMs - minLatency) / (maxLatency - minLatency), 0, 1)
}

function radiusMixFromLatencyMs(
  node: TopologyNode,
  minLatency: number,
  maxLatency: number,
  radiusSeed: number,
  radialBias: number
) {
  const latencyNorm = Math.pow(normalizedLatency(node, minLatency, maxLatency), 0.9)
  return clamp(latencyNorm * 0.78 + radiusSeed * 0.22 + radialBias, 0, 1)
}

export function distributeLatencyBand(
  nodes: TopologyNode[],
  minLatency: number,
  maxLatency: number,
  angleStart: number,
  angleEnd: number,
  innerRadiusX: number,
  outerRadiusX: number,
  innerRadiusY: number,
  outerRadiusY: number,
  armCount = 3,
  radialBias = 0,
  curveLimit = 0.7
) {
  const angleSpan = angleEnd - angleStart

  return nodes.map((node) => {
    const latencyNorm = normalizedLatency(node, minLatency, maxLatency)
    const identity = `${node.id}:${node.hostname ?? ''}:${node.serving}`
    const armSeed = hashString(`${identity}:arm`)
    const armIndex = Math.floor(armSeed * armCount) % armCount
    const armBandStart = angleStart + (armIndex / armCount) * angleSpan
    const armBandEnd = angleStart + ((armIndex + 1) / armCount) * angleSpan
    const angleSeed = hashString(`${identity}:angle`)
    const angle = armBandStart + angleSeed * (armBandEnd - armBandStart)
    const radiusSeed = hashString(`${identity}:radius`)
    const radiusMix = radiusMixFromLatencyMs(node, minLatency, maxLatency, radiusSeed, radialBias)
    const radiusX = innerRadiusX + (outerRadiusX - innerRadiusX) * radiusMix
    const radiusY = innerRadiusY + (outerRadiusY - innerRadiusY) * radiusMix
    const tangentDrift = (hashString(`${identity}:tangent-drift`) - 0.5) * Math.min(0.05, curveLimit * 0.08)
    const radialDrift = (hashString(`${identity}:radial-drift`) - 0.5) * 0.024
    const positionAtAngle = (nextAngle: number) => {
      const tangentAngle = nextAngle + Math.PI / 2
      const driftX = Math.cos(tangentAngle) * tangentDrift + Math.cos(nextAngle) * radialDrift
      const driftY = Math.sin(tangentAngle) * tangentDrift + Math.sin(nextAngle) * radialDrift
      return {
        x: clamp(0.5 + Math.cos(nextAngle) * radiusX + driftX, 0.12, 0.88),
        y: clamp(0.52 + Math.sin(nextAngle) * radiusY + driftY, 0.16, 0.84)
      }
    }
    const { x, y } = positionAtAngle(angle)

    return {
      node,
      latencyNorm,
      x,
      y,
      angle,
      positionAtAngle
    }
  })
}

export function nodeSize(node: TopologyNode, emphasis: number) {
  const base = node.client ? 8 : 10
  const vramBoost = node.client ? 0 : Math.sqrt(Math.max(0, node.vram)) * 1.55
  const maxBandwidthGbps = Math.max(0, ...(node.gpus?.map((gpu) => gpu.bandwidth_gbps ?? 0) ?? []))
  const bandwidthBoost = node.client || maxBandwidthGbps <= 0 ? 0 : clamp(Math.sqrt(maxBandwidthGbps) / 20, 0, 1.25)
  return clamp(base + vramBoost + bandwidthBoost + emphasis, node.client ? 6 : 10, 38)
}

/**
 * Iterative repulsion-based overlap resolver.
 *
 * Instead of a single-pass sampling approach, this runs a bounded force
 * relaxation inspired by ECharts/D3 force layouts:
 *   1. Pairwise repulsion pushes overlapping nodes apart (magnitude ∝ overlap).
 *   2. A gentle restoration force pulls each node back toward its original
 *      band-placed position so the overall layout shape is preserved.
 *   3. Gravity toward the center node, weighted by VRAM (node size).
 *      High-VRAM backbone nodes get strong pull; clients get near-zero.
 *   4. Exponential cooling (friction *= COOLING_RATE) converges smoothly.
 *
 * Fixed nodes (e.g. the self/center node) exert repulsion but do not move.
 */
