import type { MeshModel } from '@/features/app-shell/lib/status-types'

export function visionBadge(model?: MeshModel | null) {
  if (!model) return null
  if (model.vision) return { icon: '👁', title: 'Vision — understands images' }
  if (model.vision_status === 'likely') {
    return {
      icon: '👁?',
      title: 'Vision likely — inferred from model metadata'
    }
  }
  return null
}

export function multimodalBadge(model?: MeshModel | null) {
  if (!model) return null
  if (model.multimodal) {
    return { icon: '🎛️', title: 'Multimodal — supports media inputs' }
  }
  return null
}

export function audioBadge(model?: MeshModel | null) {
  if (!model) return null
  if (model.audio) return { icon: '🔊', title: 'Audio — understands audio input' }
  if (model.audio_status === 'likely') {
    return {
      icon: '🔊?',
      title: 'Audio likely — inferred from model metadata'
    }
  }
  return null
}

export function reasoningBadge(model?: MeshModel | null) {
  if (!model) return null
  if (model.reasoning) return { icon: '🧠', title: 'Reasoning-oriented model' }
  if (model.reasoning_status === 'likely') {
    return {
      icon: '🧠?',
      title: 'Reasoning likely — inferred from model metadata'
    }
  }
  return null
}
