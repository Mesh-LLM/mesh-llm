import type { Dispatch, SetStateAction } from 'react'
import { ModelConfigCard } from '@/features/configuration/components/ModelConfigCard'
import { ReservedConfigCard } from '@/features/configuration/components/ReservedConfigCard'
import {
  containerAssigns,
  containerReservedGB,
  containerTotalGB,
  containerUsedGB,
  findModel,
  modelWeightsGB
} from '@/features/configuration/lib/config-math'
import { reservedVramSelectionId } from '@/features/configuration/lib/selection'
import type {
  ConfigAssign,
  ConfigModel,
  ConfigNode,
  ConfigurationModelPlacementOptions
} from '@/features/app-tabs/types'

type AssignedModelConfigsProps = {
  containerIdx: number
  readOnly: boolean
  selectedId?: string | null
  node: ConfigNode
  assigns: ConfigAssign[]
  models?: ConfigModel[]
  modelPlacementOptions?: ConfigurationModelPlacementOptions
  setAssigns: Dispatch<SetStateAction<ConfigAssign[]>>
  onPick: (id: string | null) => void
}

export function AssignedModelConfigs({
  containerIdx,
  readOnly,
  selectedId,
  node,
  assigns,
  models,
  modelPlacementOptions,
  setAssigns,
  onPick
}: AssignedModelConfigsProps) {
  if (readOnly) return null

  const scopedAssigns = containerAssigns(assigns, node.id, containerIdx)
  if (scopedAssigns.length === 0) return null

  const selectedAssign = scopedAssigns.find((assign) => assign.id === selectedId)
  if (!selectedAssign) return null

  const totalGB = containerTotalGB(node, containerIdx)
  const reservedGB = containerReservedGB(node, containerIdx)

  const containerFreeGBForAssign = (assign: ConfigAssign) => {
    const model = findModel(assign.modelId, models)
    return (
      totalGB -
      reservedGB -
      containerUsedGB(
        assigns.filter((item) => item.id !== assign.id),
        node.id,
        containerIdx,
        models
      ) -
      (model ? modelWeightsGB(model) : 0)
    )
  }

  return (
    <div className="space-y-2">
      <ModelConfigCard
        key={selectedAssign.id}
        assign={selectedAssign}
        node={node}
        models={models}
        modelPlacementOptions={modelPlacementOptions}
        selected
        containerFreeGB={containerFreeGBForAssign(selectedAssign)}
        controlTabIndex={-1}
        onPick={() => onPick(selectedAssign.id)}
        onCtxChange={(ctx) =>
          setAssigns((items) => items.map((item) => (item.id === selectedAssign.id ? { ...item, ctx } : item)))
        }
        onConfigChange={(config) =>
          setAssigns((items) => items.map((item) => (item.id === selectedAssign.id ? { ...item, config } : item)))
        }
        onRemove={() => {
          setAssigns((items) => items.filter((item) => item.id !== selectedAssign.id))
          onPick(null)
        }}
      />
    </div>
  )
}

type SelectedReservedConfigProps = {
  selectedId?: string | null
  node: ConfigNode
  containerIdx: number
  locationLabel: string
  reservedGB: number
}

export function SelectedReservedConfig({
  selectedId,
  node,
  containerIdx,
  locationLabel,
  reservedGB
}: SelectedReservedConfigProps) {
  if (reservedGB <= 0 || selectedId !== reservedVramSelectionId(node.id, containerIdx)) return null

  return (
    <ReservedConfigCard
      key={`reserved-${node.id}-${containerIdx}`}
      locationLabel={locationLabel}
      reservedGB={reservedGB}
    />
  )
}
