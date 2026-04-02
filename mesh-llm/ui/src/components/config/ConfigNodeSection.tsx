import { useEffect, useMemo } from 'react';
import { TriangleAlert } from 'lucide-react';

import type { OwnedNode } from '../../hooks/useOwnedNodes';
import type { AggregatedModel } from '../../lib/models';
import { cn } from '../../lib/utils';
import type { ModelAssignment, ModelSplit, PlacementMode, ScannedModelMetadata } from '../../types/config';
import { Badge } from '../ui/badge';
import { Button } from '../ui/button';
import { Tabs, TabsList, TabsTrigger } from '../ui/tabs';
import { ModelDetailPanel } from './ModelDetailPanel';
import {
  VramContainer,
  type VramAssignment,
} from './VramContainer';

function isEditableTarget(target: EventTarget | null): boolean {
  if (!(target instanceof HTMLElement)) {
    return false;
  }

  if (target.isContentEditable) {
    return true;
  }

  const tagName = target.tagName;
  if (tagName === 'INPUT' || tagName === 'TEXTAREA' || tagName === 'SELECT') {
    return true;
  }

  return target.closest('[contenteditable="true"], [role="textbox"]') != null;
}

/** Extract the GPU ordinal from an assignment id suffix like `::gpu-0`. */
function gpuIndexFromAssignmentId(id: string): number | null {
  const match = id.match(/::gpu-(\d+)$/);
  return match ? Number.parseInt(match[1], 10) : null;
}

function splitStatusToneClass(statusTone: OwnedNode['statusTone']) {
  switch (statusTone) {
    case 'serving':
      return 'border-emerald-500/30 bg-emerald-500/8 text-emerald-700 dark:text-emerald-300';
    case 'host':
      return 'border-primary/30 bg-primary/10 text-primary';
    case 'worker':
      return 'border-sky-500/30 bg-sky-500/10 text-sky-700 dark:text-sky-300';
    case 'client':
      return 'border-border bg-muted text-muted-foreground';
    default:
      return 'border-border bg-muted/50 text-muted-foreground';
  }
}

type ConfigNodeSectionProps = {
  node: OwnedNode;
  isSelected: boolean;
  placementMode?: PlacementMode;
  totalVramBytes: number;
  assignments: VramAssignment[];
  selectedAssignmentId: string | null;
  selectedAssignmentIds: string[];
  selectedAssignment: ModelAssignment | null;
  selectedAggregated: AggregatedModel | null;
  selectedMetadata: ScannedModelMetadata | null;
  selectedGroupId: string | null;
  recombineError: string | null;
  modelScansLookup?: Map<string, ScannedModelMetadata>;
  modelKeyLookup?: Map<string, string>;
  availableNodeCount: number;
  crossNodeSplitGroupIds?: Set<string>;
  onSelectNode: (nodeId: string) => void;
  onClearSelectedAssignment: () => void;
  onRemoveModel: (modelName: string, assignmentId: string) => void;
  onSelectAssignment: (assignmentId: string) => void;
  onSplitModel: (
    modelName: string,
    blockA: { model_key: string; split: ModelSplit },
    blockB: { model_key: string; split: ModelSplit },
  ) => void;
  onRecombineGroup: (groupId: string) => void;
  onResizeSplitBoundary: (
    leftAssignmentId: string,
    rightAssignmentId: string,
    boundaryStart: number,
  ) => void;
  onUpdateModel: (assignmentId: string, updates: Partial<ModelAssignment>) => void;
  onPlacementModeChange?: (mode: PlacementMode) => void;
  onBeginBatch?: () => void;
  onEndBatch?: () => void;
};

export function ConfigNodeSection({
  node,
  isSelected,
  placementMode,
  totalVramBytes,
  assignments,
  selectedAssignmentId,
  selectedAssignmentIds,
  selectedAssignment,
  selectedAggregated,
  selectedMetadata,
  selectedGroupId,
  recombineError,
  modelScansLookup,
  modelKeyLookup,
  availableNodeCount,
  crossNodeSplitGroupIds,
  onSelectNode,
  onClearSelectedAssignment,
  onRemoveModel,
  onSelectAssignment,
  onSplitModel,
  onRecombineGroup,
  onResizeSplitBoundary,
  onUpdateModel,
  onPlacementModeChange,
  onBeginBatch,
  onEndBatch,
}: ConfigNodeSectionProps) {
  const compatibilityNode = node as OwnedNode & {
    hardwareLabel?: 'GPU' | 'SoC';
    hardwareNames?: string[];
    models?: string[];
  };
  const hardwareLabel = compatibilityNode.hardwareLabel ?? 'GPU';
  const hardwareNames = compatibilityNode.hardwareNames ?? [];
  const modelNames = compatibilityNode.models ?? [];

  const effectiveMode: PlacementMode = placementMode ?? 'pooled';
  const gpuTargets = node.gpuTargets ?? [];
  const hasMixedGpus = node.mixedGpuWarning ?? false;
  const totalReservedBytes = gpuTargets.reduce((sum, g) => sum + g.reservedBytes, 0);

  const showMixedGpuWarning = useMemo(() => {
    if (!hasMixedGpus) return false;
    if (effectiveMode === 'pooled') return true;
    // In separate mode, only warn when a model is split across GPUs
    const modelGpus = new Map<string, Set<number>>();
    for (const a of assignments) {
      const idx = gpuIndexFromAssignmentId(a.id);
      if (idx == null) continue;
      if (!modelGpus.has(a.name)) modelGpus.set(a.name, new Set());
      modelGpus.get(a.name)!.add(idx);
    }
    for (const gpus of modelGpus.values()) {
      if (gpus.size > 1) return true;
    }
    return false;
  }, [hasMixedGpus, effectiveMode, assignments]);

  const assignmentsByGpu = useMemo(() => {
    if (effectiveMode !== 'separate') return new Map<number, VramAssignment[]>();
    const map = new Map<number, VramAssignment[]>();
    for (const gpu of gpuTargets) {
      map.set(gpu.index, []);
    }
    for (const assignment of assignments) {
      const idx = gpuIndexFromAssignmentId(assignment.id);
      if (idx != null && map.has(idx)) {
        map.get(idx)!.push(assignment);
      }
    }
    return map;
  }, [assignments, effectiveMode, gpuTargets]);

  useEffect(() => {
    if (!selectedAssignment) {
      return;
    }

    const handlePointerDown = (event: PointerEvent) => {
      if (!(event.target instanceof Element)) {
        return;
      }

      if (event.target.closest('[data-config-assignment-interactive="true"]')) {
        return;
      }

      onClearSelectedAssignment();
    };

    document.addEventListener('pointerdown', handlePointerDown, true);
    return () => {
      document.removeEventListener('pointerdown', handlePointerDown, true);
    };
  }, [onClearSelectedAssignment, selectedAssignment]);

  useEffect(() => {
    if (!selectedAssignment || !selectedAssignmentId) {
      return;
    }

    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.defaultPrevented) {
        return;
      }

      if (event.key !== 'Delete' && event.key !== 'Backspace') {
        return;
      }

      if (event.metaKey || event.ctrlKey || event.altKey || event.shiftKey) {
        return;
      }

      if (isEditableTarget(event.target)) {
        return;
      }

      event.preventDefault();
      onRemoveModel(selectedAssignment.name, selectedAssignmentId);
    };

    document.addEventListener('keydown', handleKeyDown, true);
    return () => {
      document.removeEventListener('keydown', handleKeyDown, true);
    };
  }, [onRemoveModel, selectedAssignment, selectedAssignmentId]);

  return (
    <section
      data-testid={`config-node-section-${node.id}`}
      className={cn(
        'select-none rounded-xl border bg-card/95 p-4 shadow-soft transition-colors',
        isSelected ? 'border-primary/40' : 'border-border/70',
      )}
    >
      <div className="mb-4 flex items-start justify-between gap-3">
        <button
          type="button"
          className="min-w-0 flex-1 rounded-md text-left focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
          onClick={() => onSelectNode(node.id)}
        >
          <div className="space-y-2">
            <div className="flex flex-wrap items-center gap-2">
              <h2 className="text-base font-semibold tracking-tight">{node.hostname}</h2>
              {node.isSelf ? (
                <Badge className="rounded border-border bg-muted text-[11px] text-muted-foreground">
                  This node
                </Badge>
              ) : null}
              <Badge className={cn("rounded text-[11px]", splitStatusToneClass(node.statusTone))}>
                {node.statusLabel}
              </Badge>
              {!node.statusLabel.startsWith(node.role) && (
                <Badge className="rounded border-border bg-muted/60 text-[11px] text-muted-foreground">
                  {node.role}
                </Badge>
              )}
              {isSelected ? (
                <Badge className="rounded border-primary/30 bg-primary/10 text-[11px] text-primary">
                  Catalog target
                </Badge>
              ) : null}
            </div>
            <div className="flex flex-wrap gap-2 text-xs text-muted-foreground">
              <Badge
                className="max-w-full rounded border-border/70 bg-muted/30 text-xs [overflow-wrap:anywhere]"
                title={
                  hardwareNames.length > 0 ? hardwareNames.join(', ') : undefined
                }
              >
                {hardwareLabel} · {node.gpuName}
              </Badge>
              <Badge className="rounded border-border/70 bg-muted/30 text-xs">
                VRAM · {node.vramGb.toFixed(1)} GB
              </Badge>
              {!onPlacementModeChange && (
                <Badge className="rounded border-border/70 bg-muted/30 text-xs">
                  Models · {modelNames.length > 0 ? modelNames.join(', ') : 'None advertised'}
                </Badge>
              )}
            </div>
          </div>
        </button>

        {onPlacementModeChange && node.separateCapable ? (
          <Tabs
            value={effectiveMode}
            onValueChange={(v) => onPlacementModeChange(v as PlacementMode)}
          >
            <TabsList className="h-7 shrink-0" data-config-assignment-interactive="true">
              <TabsTrigger
                value="separate"
                className="px-2.5 py-0.5 text-xs"
                data-testid={`node-${node.id}-mode-separate`}
              >
                Separate
              </TabsTrigger>
              <TabsTrigger
                value="pooled"
                className="px-2.5 py-0.5 text-xs"
                data-testid={`node-${node.id}-mode-pooled`}
              >
                Pooled
              </TabsTrigger>
            </TabsList>
          </Tabs>
        ) : null}
      </div>

      <div className="space-y-4">
       {showMixedGpuWarning ? (
           <div
             data-testid={`node-${node.id}-warning-mixed-gpu`}
             className="flex items-center gap-2 rounded border border-amber-500/40 bg-amber-500/8 px-3 py-1.5 text-xs font-medium text-amber-700 dark:text-amber-300"
           >
             <TriangleAlert className="h-3.5 w-3.5 flex-none" aria-hidden="true" />
             Mixed GPU configuration — VRAM capacities differ across devices on this node. PCIe bandwidth between devices and P2P transfer support may also limit inference throughput.
           </div>
         ) : null}

        {effectiveMode === 'separate' && gpuTargets.length > 0 ? (
          gpuTargets.map((gpu) => (
            <div key={gpu.index} data-testid={`node-${node.id}-gpu-${gpu.index}-dropzone`}>
              <VramContainer
                nodeId={`${node.id}::gpu-${gpu.index}`}
                nodeHostname={gpu.label}
                totalVramBytes={gpu.vramBytes}
                assignments={assignmentsByGpu.get(gpu.index) ?? []}
                onRemoveModel={onRemoveModel}
                selectedAssignmentId={selectedAssignmentId}
                selectedAssignmentIds={selectedAssignmentIds}
                onSelectAssignment={onSelectAssignment}
                onSplitModel={onSplitModel}
                onRecombineGroup={onRecombineGroup}
                onResizeSplitBoundary={onResizeSplitBoundary}
                modelScansLookup={modelScansLookup}
                modelKeyLookup={modelKeyLookup}
                availableNodeCount={availableNodeCount}
                crossNodeSplitGroupIds={crossNodeSplitGroupIds}
                placementTarget={`${node.id}:gpu-${gpu.index}`}
                reservedBytes={gpu.reservedBytes}
              />
            </div>
          ))
        ) : (
           <div data-testid={`node-${node.id}-pool-dropzone`}>
             <VramContainer
                nodeId={node.id}
                nodeHostname={node.hostname}
                totalVramBytes={totalVramBytes}
                assignments={assignments}
                onRemoveModel={onRemoveModel}
                selectedAssignmentId={selectedAssignmentId}
                selectedAssignmentIds={selectedAssignmentIds}
                onSelectAssignment={onSelectAssignment}
                onSplitModel={onSplitModel}
                onRecombineGroup={onRecombineGroup}
                onResizeSplitBoundary={onResizeSplitBoundary}
                modelScansLookup={modelScansLookup}
                modelKeyLookup={modelKeyLookup}
                availableNodeCount={availableNodeCount}
                crossNodeSplitGroupIds={crossNodeSplitGroupIds}
                placementTarget={`${node.id}:pooled`}
                reservedBytes={totalReservedBytes}
               />
           </div>
        )}

        {selectedGroupId ? (
          <div className="flex flex-wrap items-center gap-3 rounded-lg border border-border/70 bg-muted/20 px-3 py-2 text-xs text-muted-foreground">
            <span>
              Related split blocks stay selected together so you can move them as
              a group and recombine once they are back on one node.
            </span>
            <Button
              variant="outline"
              size="sm"
              data-config-assignment-interactive="true"
              onClick={() => onRecombineGroup(selectedGroupId)}
            >
              Recombine split
            </Button>
            {recombineError ? <span className="text-destructive">{recombineError}</span> : null}
          </div>
        ) : null}

        {selectedAssignment ? (
          <ModelDetailPanel
            assignmentId={selectedAssignmentId}
            modelName={selectedAssignment.name}
            assignment={selectedAssignment}
            aggregated={selectedAggregated}
            metadata={selectedMetadata}
            onUpdateModel={onUpdateModel}
            onBeginBatch={onBeginBatch}
            onEndBatch={onEndBatch}
          />
        ) : null}
      </div>
    </section>
  );
}
