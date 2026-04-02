import { useDraggable, useDroppable, useDragOperation } from '@dnd-kit/react';
import { TriangleAlert } from 'lucide-react';
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';

import type { ModelSplit, ScannedModelMetadata } from '../../types/config';
import { cn } from '../../lib/utils';

import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from '../ui/tooltip';
import { DRAG_INTERACTIVE_ATTRIBUTE, VRAM_DROP_TARGET_PREFIX } from './DndContext';
import { BlockContextMenu, type ContextMenuPosition } from './BlockContextMenu';

export type VramAssignment = {
  id: string;
  name: string;
  sizeBytes: number;
  fullSizeBytes: number;
  weightsBytes: number;
  contextBytes: number;
  sizeGb: number;
  moeExperts?: number;
  ctxSize?: number;
  errorMessage?: string;
  invalidMessage?: string;
  model_key?: string | null;
  split?: ModelSplit | null;
};

type VramContainerProps = {
  nodeId: string;
  nodeHostname: string;
  totalVramBytes: number;
  assignments: VramAssignment[];
  onRemoveModel: (modelName: string, assignmentId: string) => void;
  selectedAssignmentId?: string | null;
  selectedAssignmentIds?: string[];
  onSelectAssignment: (assignmentId: string) => void;
  onSplitModel?: (modelName: string, blockA: { model_key: string; split: ModelSplit }, blockB: { model_key: string; split: ModelSplit }) => void;
  onRecombineGroup?: (groupId: string) => void;
  onResizeSplitBoundary?: (leftAssignmentId: string, rightAssignmentId: string, boundaryStart: number) => void;
  modelScansLookup?: Map<string, ScannedModelMetadata>;
  modelKeyLookup?: Map<string, string>;
  availableNodeCount?: number;
  placementTarget?: string;
  crossNodeSplitGroupIds?: Set<string>;
  reservedBytes?: number;
};

type FitStatus = 'fits' | 'overcommit' | 'unknown';

type ResizePreview = {
  leftAssignmentId: string;
  rightAssignmentId: string;
  boundaryStart: number;
} | null;

type DisplayAssignment = VramAssignment & {
  displaySplit: ModelSplit | null;
  displaySizeBytes: number;
};

type AssignmentRun = {
  key: string;
  groupId: string | null;
  assignments: DisplayAssignment[];
  totalBytes: number;
};

function formatGb(bytes: number) {
  const gb = bytes / 1e9;
  return `${gb >= 100 ? Math.round(gb) : gb.toFixed(1)} GB`;
}

/** Like formatGb but switches to MB when the value is small (< 2 GB). */
function formatVramCompact(bytes: number) {
  const gb = bytes / 1e9;
  if (gb < 2) {
    return `${Math.round(bytes / 1e6)} MB`;
  }
  return `${gb >= 100 ? Math.round(gb) : gb.toFixed(1)} GB`;
}

function computeFitStatus(assignedBytes: number, totalVramBytes: number): FitStatus {
  if (totalVramBytes <= 0) return 'unknown';
  return assignedBytes <= totalVramBytes ? 'fits' : 'overcommit';
}

function fitStatusColor(status: FitStatus) {
  if (status === 'overcommit') return 'bg-destructive/15 border-destructive/30 text-destructive';
  if (status === 'fits') return 'bg-primary/8 border-primary/20 text-foreground';
  return 'bg-muted/30 border-border/40 text-foreground';
}

function FitDot({ status }: { status: FitStatus }) {
  return (
    <span
      data-testid={`fit-dot-${status}`}
      className={cn(
        'inline-block h-1.5 w-1.5 flex-none rounded-full',
        status === 'fits' && 'bg-emerald-500',
        status === 'overcommit' && 'bg-destructive',
        status === 'unknown' && 'bg-muted-foreground/40',
      )}
    />
  );
}

function modelColorHue(name: string): number {
  let hash = 5381;
  for (let i = 0; i < name.length; i++) {
    hash = ((hash << 5) + hash + name.charCodeAt(i)) | 0;
  }
  return ((hash % 360) + 360) % 360;
}

function modelBlockStyle(modelName: string) {
  const hue = modelColorHue(modelName);
  return {
    borderColor: `oklch(0.55 0.15 ${hue} / 0.35)`,
  };
}

function selectedModelBlockStyle(modelName: string) {
  const hue = modelColorHue(modelName);
  return {
    borderColor: `oklch(0.78 0.2 ${hue} / 0.85)`,
    boxShadow: `inset 0 0 0 1px oklch(0.84 0.18 ${hue} / 0.65), 0 0 0 1px oklch(0.74 0.18 ${hue} / 0.35)`,
  };
}

function scaleBytes(bytes: number, ratio: number) {
  return Math.round(bytes * ratio);
}

function splitGroupId(assignment: Pick<VramAssignment, 'name' | 'model_key' | 'split'>): string | null {
  if (!assignment.split || !assignment.model_key) return null;
  return `${assignment.name}::${assignment.model_key}::${assignment.split.total}`;
}

function splitLayerCount(split: ModelSplit) {
  return split.end - split.start + 1;
}

function canResizeBetween(left: Pick<VramAssignment, 'name' | 'model_key' | 'split'>, right: Pick<VramAssignment, 'name' | 'model_key' | 'split'>) {
  return splitGroupId(left) != null
    && splitGroupId(left) === splitGroupId(right)
    && left.split != null
    && right.split != null
    && left.split.end + 1 === right.split.start;
}

function toDisplayAssignment(assignment: VramAssignment, preview: ResizePreview): DisplayAssignment {
  if (!assignment.split) {
    return {
      ...assignment,
      displaySplit: assignment.split ?? null,
      displaySizeBytes: assignment.sizeBytes,
    };
  }

  let displaySplit = assignment.split;

  if (preview && assignment.id === preview.leftAssignmentId) {
    displaySplit = { ...assignment.split, end: preview.boundaryStart - 1 };
  } else if (preview && assignment.id === preview.rightAssignmentId) {
    displaySplit = { ...assignment.split, start: preview.boundaryStart };
  }

  return {
    ...assignment,
    displaySplit,
    displaySizeBytes: scaleBytes(assignment.fullSizeBytes, splitLayerCount(displaySplit) / displaySplit.total),
    weightsBytes: scaleBytes(assignment.weightsBytes, splitLayerCount(displaySplit) / displaySplit.total),
    contextBytes: scaleBytes(assignment.contextBytes, splitLayerCount(displaySplit) / displaySplit.total),
  };
}

function resolveOffloadableLayers(
  assignment: VramAssignment,
  scansLookup?: Map<string, ScannedModelMetadata>,
): number | null {
  const meta = scansLookup?.get(assignment.name);
  if (meta?.total_offloadable_layers && meta.total_offloadable_layers > 0) {
    return meta.total_offloadable_layers;
  }
  return null;
}

function getSplitActionState(
  assignment: VramAssignment,
  scansLookup?: Map<string, ScannedModelMetadata>,
  modelKeyLookup?: Map<string, string>,
  availableNodeCount?: number,
): { canSplit: boolean; reason: string | null } {
  if (assignment.split) {
    return { canSplit: false, reason: 'This block is already split.' };
  }
  if (typeof availableNodeCount === 'number' && availableNodeCount <= 1) {
    return { canSplit: false, reason: 'Split unavailable: add another available node to distribute this model.' };
  }

  const modelKey = assignment.model_key ?? findModelKey(assignment.name, modelKeyLookup);
  const totalLayers = resolveOffloadableLayers(assignment, scansLookup);
  if (!totalLayers || totalLayers <= 0) {
    return { canSplit: false, reason: 'Split unavailable: offloadable layer scan metadata is missing for this node.' };
  }
  if (totalLayers <= 1) {
    return { canSplit: false, reason: 'Split unavailable: this model only exposes one offloadable layer.' };
  }
  if (!modelKey) {
    return { canSplit: false, reason: 'Split unavailable: matching scan metadata is missing for this node.' };
  }

  return { canSplit: true, reason: null };
}

function buildAssignmentRuns(assignments: VramAssignment[], preview: ResizePreview): AssignmentRun[] {
  const displayAssignments = assignments.map((assignment) => toDisplayAssignment(assignment, preview));
  const displayById = new Map(displayAssignments.map((assignment) => [assignment.id, assignment]));

  const runs: AssignmentRun[] = [];

  for (const assignment of assignments) {
    const displayAssignment = displayById.get(assignment.id);
    if (!displayAssignment) continue;

    const groupId = splitGroupId(assignment);
    const previous = runs[runs.length - 1];
    if (groupId && previous?.groupId === groupId) {
      previous.assignments.push(displayAssignment);
      previous.totalBytes += displayAssignment.displaySizeBytes;
      continue;
    }

    runs.push({
      key: groupId ? `group:${groupId}:${runs.length}` : `single:${assignment.id}`,
      groupId,
      assignments: [displayAssignment],
      totalBytes: displayAssignment.displaySizeBytes,
    });
  }

  return runs;
}

export function VramContainer({
  nodeId,
  nodeHostname,
  totalVramBytes,
  assignments,
  onRemoveModel,
  selectedAssignmentId,
  selectedAssignmentIds,
  onSelectAssignment,
  onSplitModel,
  onRecombineGroup,
  onResizeSplitBoundary,
  modelScansLookup,
  modelKeyLookup,
  availableNodeCount,
  placementTarget,
  crossNodeSplitGroupIds,
  reservedBytes = 0,
}: VramContainerProps) {
  const assignedBytes = assignments.reduce((sum, assignment) => sum + assignment.sizeBytes, 0);
  const freeBytes = Math.max(0, totalVramBytes - assignedBytes);
  const overcommitBytes = Math.max(0, assignedBytes - totalVramBytes);
  const isOvercommitted = overcommitBytes > 0;

  const reservedPercent = totalVramBytes > 0 ? (reservedBytes / totalVramBytes) * 100 : 0;

  const { ref, isDropTarget } = useDroppable({
    id: `${VRAM_DROP_TARGET_PREFIX}${nodeId}`,
    data: placementTarget != null ? { placementTarget } : undefined,
    disabled: isOvercommitted,
  });
  const { source } = useDragOperation();

  const containerRef: React.MutableRefObject<HTMLDivElement | null> = useRef(null);
  const mergedContainerRef = useCallback(
    (el: HTMLDivElement | null) => {
      ref(el);
      containerRef.current = el;
    },
    [ref],
  );
  const [contextMenu, setContextMenu] = useState<{ position: ContextMenuPosition; assignment: VramAssignment } | null>(null);
  const [resizePreview, setResizePreview] = useState<ResizePreview>(null);
  const selectedAssignmentIdSet = useMemo(
    () => new Set(selectedAssignmentIds ?? (selectedAssignmentId ? [selectedAssignmentId] : [])),
    [selectedAssignmentId, selectedAssignmentIds],
  );

  const displayRuns = useMemo(() => buildAssignmentRuns(assignments, resizePreview), [assignments, resizePreview]);

  const dragData = source?.data as { type?: string; sizeBytes?: number } | undefined;
  const isDraggingAssignable = isDropTarget
    && (dragData?.type === 'model' || dragData?.type === 'split-assignment')
    && typeof dragData.sizeBytes === 'number';
  const isDraggingSplitAssignment = isDraggingAssignable && dragData?.type === 'split-assignment';
  const draggedSizeBytes = isDraggingAssignable ? (dragData.sizeBytes ?? 0) : 0;
  const wouldFit = isDraggingAssignable ? assignedBytes + draggedSizeBytes * 1.1 <= totalVramBytes : null;

  const closeMenu = useCallback(() => setContextMenu(null), []);

  const handleContextMenu = useCallback((e: React.MouseEvent, assignment: VramAssignment) => {
    e.preventDefault();
    e.stopPropagation();
    setContextMenu({ position: { x: e.clientX, y: e.clientY }, assignment });
  }, []);

  const handleSplit = useCallback((assignment: VramAssignment) => {
    if (!onSplitModel) return;

    const modelKey = assignment.model_key ?? findModelKey(assignment.name, modelKeyLookup);
    if (!modelKey) return;

    const totalLayers = resolveOffloadableLayers(assignment, modelScansLookup);
    if (!totalLayers || totalLayers <= 0) return;

    const half = Math.floor(totalLayers / 2);
    const blockA = { model_key: modelKey, split: { start: 0, end: half - 1, total: totalLayers } };
    const blockB = { model_key: modelKey, split: { start: half, end: totalLayers - 1, total: totalLayers } };

    onSplitModel(assignment.name, blockA, blockB);
    closeMenu();
  }, [closeMenu, modelKeyLookup, modelScansLookup, onSplitModel]);

  const handleRemove = useCallback((assignment: VramAssignment) => {
    onRemoveModel(assignment.name, assignment.id);
    closeMenu();
  }, [closeMenu, onRemoveModel]);

  const handleRecombine = useCallback((assignment: VramAssignment) => {
    const groupId = splitGroupId(assignment);
    if (!groupId || !onRecombineGroup) return;
    onRecombineGroup(groupId);
    closeMenu();
  }, [closeMenu, onRecombineGroup]);

  const usedPercent = totalVramBytes > 0 ? Math.min((assignedBytes / totalVramBytes) * 100, 100) : 0;
  const fitStatus = computeFitStatus(assignedBytes, totalVramBytes);
  const freePercent = totalVramBytes > 0 ? Math.max(0, 100 - (assignedBytes / totalVramBytes) * 100) : 0;
  const barFreePercent = Math.max(0, freePercent - reservedPercent);
  const splitActionState = contextMenu
    ? getSplitActionState(contextMenu.assignment, modelScansLookup, modelKeyLookup, availableNodeCount)
    : null;

  const handleKeyDown = useCallback((event: React.KeyboardEvent<HTMLDivElement>) => {
    if (event.key !== 'ArrowDown' && event.key !== 'ArrowUp' && event.key !== 'Enter' && event.key !== ' ') {
      return;
    }

    const modelBlocks = containerRef.current?.querySelectorAll('[data-testid="vram-model-block"]') as NodeListOf<HTMLButtonElement> | undefined;
    if (!modelBlocks || modelBlocks.length === 0) return;

    const activeElement = document.activeElement as HTMLElement | null;
    const focusedBlockIndex = Array.from(modelBlocks).findIndex((el) => el === activeElement);

    if (event.key === 'ArrowDown') {
      event.preventDefault();
      const nextIndex = focusedBlockIndex < 0 ? 0 : Math.min(focusedBlockIndex + 1, modelBlocks.length - 1);
      modelBlocks[nextIndex]?.focus();
    } else if (event.key === 'ArrowUp') {
      event.preventDefault();
      const prevIndex = focusedBlockIndex <= 0 ? 0 : focusedBlockIndex - 1;
      modelBlocks[prevIndex]?.focus();
    } else if ((event.key === 'Enter' || event.key === ' ') && focusedBlockIndex >= 0) {
      event.preventDefault();
      modelBlocks[focusedBlockIndex]?.click();
    }
  }, []);

  return (
    <TooltipProvider>
      <section
        ref={mergedContainerRef}
        data-testid="vram-container"
        className={cn(
          'rounded-lg border p-4 transition-all duration-200',
          isOvercommitted
            ? 'border-destructive/40 bg-destructive/5 shadow-[inset_0_1px_0_0_hsl(0_84%_60%/0.08)]'
            : isDraggingAssignable
              ? wouldFit
                ? 'border-emerald-500/50 bg-emerald-500/5 ring-1 ring-emerald-500/20'
                : 'border-destructive/40 bg-destructive/5 ring-1 ring-destructive/15'
              : 'border-border/60 bg-muted/15 shadow-[inset_0_1px_0_0_hsl(var(--foreground)/0.03)]',
        )}
        data-placement-target={placementTarget}
        onKeyDown={handleKeyDown}
        aria-label="VRAM container with keyboard navigation"
      >
      <div className="mb-3 flex flex-wrap items-baseline justify-between gap-x-4 gap-y-1">
        <div className="text-sm font-semibold tracking-tight text-foreground">
          <span className="text-muted-foreground">VRAM</span>{' '}
          <span>{nodeHostname}</span>
        </div>
        <div className="flex gap-3 text-xs text-muted-foreground font-mono">
          <span>
            Total <span className="font-medium text-foreground">{formatGb(totalVramBytes)}</span>
          </span>
          <span>
            Used <span className="font-medium text-foreground">{formatGb(assignedBytes)}</span>
          </span>
          <span>
            Free{' '}
            <span className={cn('font-medium', isOvercommitted ? 'text-destructive' : 'text-foreground')}>
              {isOvercommitted ? `-${formatGb(overcommitBytes)}` : formatVramCompact(freeBytes)}
            </span>
          </span>
        </div>
      </div>

      <div className="mb-3 h-2 overflow-hidden rounded-full bg-muted/40" title={`${usedPercent.toFixed(0)}% VRAM used`}>
        <div
          className={cn(
            'h-full rounded-full transition-[width] duration-200',
            isOvercommitted ? 'bg-destructive' : 'bg-primary/60',
          )}
          style={{ width: `${usedPercent}%` }}
        />
      </div>

      {isOvercommitted ? (
        <div
          data-testid="vram-overcommit-warning"
          className="mb-3 rounded border border-destructive/50 bg-destructive/10 px-3 py-1.5 text-xs font-medium text-destructive"
        >
          Overcommitted by {formatGb(overcommitBytes)}
        </div>
      ) : null}

      {assignments.length > 0 || reservedBytes > 0 ? (
        <div className="flex min-h-[3rem] w-full gap-1.5 overflow-visible rounded-md border border-border/40" data-testid="vram-capacity-bar">
          {reservedPercent > 0.5 ? (
            <Tooltip>
              <TooltipTrigger asChild>
                <div
                  data-testid="vram-reserved-block"
                  className="flex flex-col items-center justify-center overflow-hidden text-[9px] font-medium text-muted-foreground/60"
                  style={{
                    flexBasis: `${reservedPercent}%`,
                    flexGrow: 0,
                    flexShrink: 0,
                    minWidth: '1.5rem',
                    backgroundColor: 'hsl(0 0% 50% / 0.10)',
                    backgroundImage:
                      'repeating-linear-gradient(-45deg, transparent, transparent 4px, hsl(0 0% 50% / 0.25) 4px, hsl(0 0% 50% / 0.25) 5px)',
                  }}
                >
                  sys
                </div>
              </TooltipTrigger>
              <TooltipContent side="top" className="text-xs">
                System reserved · {formatVramCompact(reservedBytes)}
              </TooltipContent>
            </Tooltip>
          ) : null}

          {displayRuns.map((run) => {
            const widthPercent = totalVramBytes > 0 ? (run.totalBytes / totalVramBytes) * 100 : 100 / displayRuns.length;

             if (run.groupId && run.assignments.length > 1) {
               return (
                 <SplitGroupBlock
                   key={run.key}
                   nodeId={nodeId}
                   assignments={run.assignments}
                   fitStatus={fitStatus}
                   groupWidthPercent={widthPercent}
                   selectedAssignmentIdSet={selectedAssignmentIdSet}
                   activeDragId={String(source?.id ?? '')}
                   onContextMenu={handleContextMenu}
                   onSelectAssignment={onSelectAssignment}
                   onPreviewResize={(leftAssignmentId, rightAssignmentId, boundaryStart) => {
                     setResizePreview({ leftAssignmentId, rightAssignmentId, boundaryStart });
                   }}
                   onCommitResize={(leftAssignmentId, rightAssignmentId, boundaryStart) => {
                     setResizePreview(null);
                     onResizeSplitBoundary?.(leftAssignmentId, rightAssignmentId, boundaryStart);
                   }}
                   isCrossNodeSplit={run.groupId ? crossNodeSplitGroupIds?.has(run.groupId) ?? false : false}
                 />
               );
             }

            const assignment = run.assignments[0];
            return (
              <ModelBlock
                key={assignment.id}
                nodeId={nodeId}
                assignment={assignment}
                fitStatus={fitStatus}
                isSelected={selectedAssignmentIdSet.has(assignment.id)}
                activeDragId={String(source?.id ?? '')}
                onSelectAssignment={onSelectAssignment}
                onContextMenu={handleContextMenu}
                widthPercent={widthPercent}
                hasRightBorder={false}
              />
            );
          })}

          {!isOvercommitted && barFreePercent > 1 ? (
            <div
              data-testid="vram-free-block"
              className="flex flex-col items-center justify-center bg-muted/10 text-[9px] text-muted-foreground/40"
              style={{ flexBasis: `${barFreePercent}%`, flexGrow: 0, flexShrink: 1, minWidth: '2rem' }}
            >
              free
            </div>
          ) : null}
        </div>
      ) : null}

      {assignments.length === 0 ? (
        <div
          data-testid="vram-empty"
          className={cn(
            reservedBytes > 0
              ? 'mt-2 flex min-h-[2rem] items-center justify-center rounded-md border text-xs text-muted-foreground/50'
              : 'flex min-h-[4rem] items-center justify-center rounded-md border text-xs text-muted-foreground/60',
            isDraggingAssignable
              ? wouldFit
                ? 'border-emerald-500/40 bg-emerald-500/5 text-emerald-600/60 dark:text-emerald-400/60'
                : 'border-destructive/40 bg-destructive/5 text-destructive/60'
              : 'border-border/30 bg-muted/5',
          )}
        >
          Drag models from the catalog to assign them
        </div>
      ) : null}

      {isDraggingAssignable ? (
        <div
          data-testid="vram-drag-preview"
          className={cn(
            'mt-2 rounded px-2 py-1 text-center text-xs font-medium',
            wouldFit ? 'bg-emerald-500/10 text-emerald-600 dark:text-emerald-300' : 'bg-destructive/10 text-destructive',
          )}
        >
          {wouldFit
            ? isDraggingSplitAssignment
              ? 'Drop to move split'
              : 'Drop to assign'
            : 'Not enough VRAM'}
        </div>
      ) : null}

        {contextMenu ? (
          <BlockContextMenu
            position={contextMenu.position}
            assignment={contextMenu.assignment}
            canSplit={splitActionState?.canSplit ?? false}
            splitReason={splitActionState?.reason ?? null}
            onRecombine={contextMenu.assignment.split ? () => handleRecombine(contextMenu.assignment) : null}
            onSplit={() => handleSplit(contextMenu.assignment)}
            onRemove={() => handleRemove(contextMenu.assignment)}
            onClose={closeMenu}
          />
        ) : null}
      </section>
    </TooltipProvider>
  );
}

function ModelBlock({
  nodeId,
  assignment,
  fitStatus,
  isSelected,
  activeDragId,
  onSelectAssignment,
  onContextMenu,
  widthPercent,
  hasRightBorder,
}: {
  nodeId: string;
  assignment: DisplayAssignment;
  fitStatus: FitStatus;
  isSelected: boolean;
  activeDragId: string;
  onSelectAssignment: (assignmentId: string) => void;
  onContextMenu: (event: React.MouseEvent, assignment: VramAssignment) => void;
  widthPercent: number;
  hasRightBorder: boolean;
}) {
  const dragId = `split-block:${assignment.id}`;
  const { ref, handleRef } = useDraggable({
    id: dragId,
    disabled: !assignment.displaySplit,
    data: {
      type: 'split-assignment',
      assignmentId: assignment.id,
      sourceNodeId: nodeId,
      modelName: assignment.name,
      sizeBytes: assignment.displaySizeBytes,
    },
  });

  const splitLabel = assignment.displaySplit ? `L${assignment.displaySplit.start}–${assignment.displaySplit.end}` : null;
  const blockStyle = fitStatus === 'overcommit'
    ? undefined
    : isSelected
      ? selectedModelBlockStyle(assignment.name)
      : modelBlockStyle(assignment.name);
  const breakdownLabel = assignment.contextBytes > 0
    ? `${formatGb(assignment.weightsBytes)} + ${formatGb(assignment.contextBytes)} ctx`
    : formatGb(assignment.weightsBytes);
  const splitMetaTextClass = fitStatus === 'overcommit' ? 'text-destructive/90' : 'text-foreground/90';
  const breakdownTextClass = fitStatus === 'overcommit' ? 'text-destructive/80' : 'text-foreground/80';

  return (
    <div
      ref={ref}
      className={cn('relative min-w-0', hasRightBorder && 'border-r border-border/20')}
      style={{
        flexBasis: `${Math.max(widthPercent, 8)}%`,
        flexGrow: 0,
        flexShrink: 1,
        minWidth: '3.5rem',
      }}
    >
      <button
        ref={handleRef}
        type="button"
        data-testid="vram-model-block"
        data-config-assignment-interactive="true"
        data-assignment-id={assignment.id}
        tabIndex={0}
        onClick={() => onSelectAssignment(assignment.id)}
        onContextMenu={(event) => onContextMenu(event, assignment)}
        className={cn(
          'group relative flex h-full min-h-[3rem] w-full min-w-0 flex-col items-start justify-center rounded-md border px-2 py-1.5 text-left transition-all duration-150 focus-visible:ring-2 focus-visible:ring-ring focus-visible:outline-none',
          fitStatus === 'overcommit'
            ? fitStatusColor(fitStatus)
            : 'text-foreground',
          assignment.displaySplit && 'cursor-grab active:cursor-grabbing',
          activeDragId === dragId && 'opacity-80',
          assignment.invalidMessage && 'ring-1 ring-inset ring-destructive/70',
          isSelected && 'ring-1 ring-inset brightness-110',
          !isSelected && 'hover:brightness-105',
        )}
        style={blockStyle}
        title={`${assignment.name}${splitLabel ? ` (${splitLabel})` : ''} — ${breakdownLabel}${assignment.invalidMessage ? ` — ${assignment.invalidMessage}` : ''}`}
      >
        {fitStatus !== 'overcommit' && (
          <VramBreakdownBar modelName={assignment.name} weightsBytes={assignment.weightsBytes} contextBytes={assignment.contextBytes} isSelected={isSelected} />
        )}
        <div className="relative z-10 flex w-full min-w-0 flex-col items-start">
          <div className="flex w-full min-w-0 items-center gap-1">
            <FitDot status={fitStatus} />
            <span className="min-w-0 flex-1 truncate text-[11px] font-medium leading-tight">
              {assignment.name}
            </span>
            {assignment.invalidMessage ? (
              <span
                data-testid="model-block-invalid-badge"
                className="flex-none rounded bg-destructive/15 px-1 text-[9px] font-medium uppercase tracking-[0.12em] text-destructive"
              >
                invalid
              </span>
            ) : null}
            {assignment.errorMessage ? (
              <Tooltip>
                <TooltipTrigger asChild>
                  <span
                    data-testid="model-block-error"
                    className="flex h-4 w-4 flex-none items-center justify-center rounded-sm border border-destructive/60 bg-destructive text-destructive-foreground shadow-sm"
                    title="Split placement warning"
                  >
                    <TriangleAlert className="h-2.5 w-2.5" aria-hidden="true" />
                  </span>
                </TooltipTrigger>
                <TooltipContent side="top" align="center" className="max-w-64 text-pretty leading-relaxed">
                  This split has a placement warning: {assignment.errorMessage}
                </TooltipContent>
              </Tooltip>
            ) : null}
          </div>
          {splitLabel ? (
            <span data-testid="model-block-split-label" className={cn('mt-0.5 text-[9px] font-mono', splitMetaTextClass)}>
              {splitLabel}
            </span>
          ) : null}
          <span className={cn('mt-0.5 text-[9px]', breakdownTextClass)}>{breakdownLabel}</span>
          {assignment.invalidMessage ? (
            <span
              data-testid="model-block-invalid-message"
              className="mt-0.5 w-full truncate text-[9px] font-medium text-destructive"
              title={assignment.invalidMessage}
            >
              {assignment.invalidMessage}
            </span>
          ) : null}
        </div>
      </button>
    </div>
  );
}

function SplitGroupBlock({
  nodeId,
  assignments,
  fitStatus,
  groupWidthPercent,
  selectedAssignmentIdSet,
  activeDragId,
  onContextMenu,
  onSelectAssignment,
  onPreviewResize,
  onCommitResize,
  isCrossNodeSplit,
}: {
  nodeId: string;
  assignments: DisplayAssignment[];
  fitStatus: FitStatus;
  groupWidthPercent: number;
  selectedAssignmentIdSet: Set<string>;
  activeDragId: string;
  onContextMenu: (event: React.MouseEvent, assignment: VramAssignment) => void;
  onSelectAssignment: (assignmentId: string) => void;
  onPreviewResize: (leftAssignmentId: string, rightAssignmentId: string, boundaryStart: number) => void;
  onCommitResize: (leftAssignmentId: string, rightAssignmentId: string, boundaryStart: number) => void;
  isCrossNodeSplit?: boolean;
}) {
  const groupRef = useRef<HTMLDivElement>(null);
  const totalBytes = assignments.reduce((sum, assignment) => sum + assignment.displaySizeBytes, 0);
  let precedingBytes = 0;

  return (
    <div
      ref={groupRef}
      data-testid="split-group-block"
      className="flex min-w-0 items-stretch"
      style={{
        flexBasis: `${Math.max(groupWidthPercent, 8)}%`,
        flexGrow: 0,
        flexShrink: 1,
        minWidth: `${assignments.length * 3.5}rem`,
      }}
    >
      {assignments.map((assignment, index) => {
        const widthPercent = totalBytes > 0 ? (assignment.displaySizeBytes / totalBytes) * 100 : 100 / assignments.length;
        const leftPrecedingBytes = precedingBytes;
        precedingBytes += assignment.displaySizeBytes;

        const next = assignments[index + 1];

        return [
          <ModelBlock
            key={`${assignment.id}-block`}
            nodeId={nodeId}
            assignment={assignment}
            fitStatus={fitStatus}
            isSelected={selectedAssignmentIdSet.has(assignment.id)}
            activeDragId={activeDragId}
            onSelectAssignment={onSelectAssignment}
            onContextMenu={onContextMenu}
            widthPercent={widthPercent}
            hasRightBorder={false}
          />,
           next && canResizeBetween(assignment, next) ? (
             <SplitResizeHandle
               key={`${assignment.id}-handle`}
               groupRef={groupRef}
               leftAssignment={assignment}
               rightAssignment={next}
               precedingBytes={leftPrecedingBytes}
               groupTotalBytes={totalBytes}
               onPreviewBoundary={(boundaryStart) => onPreviewResize(assignment.id, next.id, boundaryStart)}
               onCommitBoundary={(boundaryStart) => onCommitResize(assignment.id, next.id, boundaryStart)}
               isCrossNodeSplit={isCrossNodeSplit}
             />
           ) : null,
        ];
      })}
    </div>
  );
}

function SplitResizeHandle({
  groupRef,
  leftAssignment,
  rightAssignment,
  precedingBytes,
  groupTotalBytes,
  onPreviewBoundary,
  onCommitBoundary,
  isCrossNodeSplit,
}: {
  groupRef: React.RefObject<HTMLDivElement | null>;
  leftAssignment: DisplayAssignment;
  rightAssignment: DisplayAssignment;
  precedingBytes: number;
  groupTotalBytes: number;
  onPreviewBoundary: (boundaryStart: number) => void;
  onCommitBoundary: (boundaryStart: number) => void;
  isCrossNodeSplit?: boolean;
}) {
  const handleRef = useRef<HTMLButtonElement>(null);
  const committedBoundaryRef = useRef<number>(rightAssignment.split?.start ?? 0);
  const dragCleanupRef = useRef<(() => void) | null>(null);
  const leftCommittedSplit = leftAssignment.split;
  const rightCommittedSplit = rightAssignment.split;
  const leftCommittedSizeBytes = leftAssignment.sizeBytes;
  const rightCommittedSizeBytes = rightAssignment.sizeBytes;

  useEffect(() => () => {
    dragCleanupRef.current?.();
  }, []);

  const resolveBoundaryPositions = useCallback(() => {
    if (!leftCommittedSplit || !rightCommittedSplit) return [] as Array<{ boundaryStart: number; px: number }>;

    const groupEl = groupRef.current;
    if (!groupEl) return [];

    const width = groupEl.getBoundingClientRect().width;
    const pairWidth = width * ((leftCommittedSizeBytes + rightCommittedSizeBytes) / Math.max(groupTotalBytes, 1));
    const pairStart = width * (precedingBytes / Math.max(groupTotalBytes, 1));
    const pairLayers = rightCommittedSplit.end - leftCommittedSplit.start + 1;

    const minBoundary = leftCommittedSplit.start + 2;
    const maxBoundary = rightCommittedSplit.end - 1;

    return Array.from({ length: rightCommittedSplit.end - leftCommittedSplit.start }, (_, index) => {
      const boundaryStart = leftCommittedSplit.start + 1 + index;
      const offset = ((boundaryStart - leftCommittedSplit.start) / pairLayers) * pairWidth;
      return { boundaryStart, px: pairStart + offset };
    }).filter(p => p.boundaryStart >= minBoundary && p.boundaryStart <= maxBoundary);
  }, [
    groupRef,
    groupTotalBytes,
    leftCommittedSizeBytes,
    leftCommittedSplit,
    precedingBytes,
    rightCommittedSizeBytes,
    rightCommittedSplit,
  ]);

  const handlePointerDown = useCallback((event: React.PointerEvent<HTMLButtonElement>) => {
    if (event.button !== 0 || !leftCommittedSplit || !rightCommittedSplit) return;

    const handleEl = handleRef.current;
    const positions = resolveBoundaryPositions();
    if (!handleEl || positions.length === 0) return;

    event.preventDefault();
    event.stopPropagation();

    dragCleanupRef.current?.();

    const startClientX = event.clientX;
    const startBoundary = rightCommittedSplit.start;
    const startHandlePx = positions.find((position) => position.boundaryStart === startBoundary)?.px ?? positions[0].px;

    committedBoundaryRef.current = startBoundary;
    onPreviewBoundary(startBoundary);

    const updateBoundary = (clientX: number) => {
      const absoluteX = startHandlePx + (clientX - startClientX);
      const next = positions.reduce((best, position) =>
        Math.abs(position.px - absoluteX) < Math.abs(best.px - absoluteX) ? position : best,
      positions[0]);

      if (next.boundaryStart !== committedBoundaryRef.current) {
        committedBoundaryRef.current = next.boundaryStart;
        onPreviewBoundary(next.boundaryStart);
      }
    };

    const cleanup = () => {
      window.removeEventListener('pointermove', handlePointerMove);
      window.removeEventListener('pointerup', handlePointerUp);
      window.removeEventListener('pointercancel', handlePointerCancel);
      if (handleEl.hasPointerCapture?.(event.pointerId)) {
        handleEl.releasePointerCapture(event.pointerId);
      }
      dragCleanupRef.current = null;
    };

    const handlePointerMove = (moveEvent: PointerEvent) => {
      updateBoundary(moveEvent.clientX);
    };

    const handlePointerUp = (upEvent: PointerEvent) => {
      updateBoundary(upEvent.clientX);
      const boundaryStart = committedBoundaryRef.current;
      cleanup();
      onCommitBoundary(boundaryStart);
    };

    const handlePointerCancel = () => {
      cleanup();
      committedBoundaryRef.current = startBoundary;
      onPreviewBoundary(startBoundary);
    };

    dragCleanupRef.current = cleanup;
    handleEl.setPointerCapture?.(event.pointerId);
    window.addEventListener('pointermove', handlePointerMove);
    window.addEventListener('pointerup', handlePointerUp);
    window.addEventListener('pointercancel', handlePointerCancel);
  }, [leftCommittedSplit, onCommitBoundary, onPreviewBoundary, resolveBoundaryPositions, rightCommittedSplit]);

  const tooltipContent = isCrossNodeSplit
    ? "Cross-node splits cannot be resized from this view. To adjust, recombine and re-split."
    : undefined;

  return (
    <div className="relative z-20 flex w-3 flex-none items-stretch justify-center">
      {isCrossNodeSplit ? (
        <Tooltip>
          <TooltipTrigger asChild>
            <button
              ref={handleRef}
              type="button"
              data-testid="split-resize-handle"
              data-config-assignment-interactive="true"
              disabled
              aria-label={`Resize split between ${leftAssignment.name} segments (disabled for cross-node splits)`}
              title={tooltipContent}
              className="my-1 w-2 flex-1 cursor-not-allowed rounded-full bg-muted/50 ring-1 ring-border/50 shadow-sm opacity-50"
            >
              <span className="sr-only">Resize split boundary (disabled)</span>
            </button>
          </TooltipTrigger>
          <TooltipContent side="top" align="center" className="max-w-xs text-pretty">
            {tooltipContent}
          </TooltipContent>
        </Tooltip>
      ) : (
        <button
          ref={handleRef}
          type="button"
          data-testid="split-resize-handle"
          data-config-assignment-interactive="true"
          aria-label={`Resize split between ${leftAssignment.name} segments`}
          {...{ [DRAG_INTERACTIVE_ATTRIBUTE]: '' }}
          onPointerDown={handlePointerDown}
          className="my-1 w-2 flex-1 cursor-col-resize rounded-full bg-background/90 ring-1 ring-border shadow-sm transition-colors hover:bg-primary/15 hover:ring-primary/40"
        >
          <span className="sr-only">Resize split boundary</span>
        </button>
      )}
    </div>
  );
}

function VramBreakdownBar({ modelName, weightsBytes, contextBytes, isSelected }: { modelName: string; weightsBytes: number; contextBytes: number; isSelected?: boolean }) {
  const totalBytes = weightsBytes + contextBytes;
  if (totalBytes <= 0) return null;

  const hue = modelColorHue(modelName);
  const weightsColor = isSelected
    ? `oklch(0.64 0.16 ${hue} / 0.32)`
    : `oklch(0.55 0.12 ${hue} / 0.2)`;
  const contextColor = isSelected
    ? `oklch(0.42 0.10 ${hue} / 0.35)`
    : `oklch(0.35 0.08 ${hue} / 0.3)`;

  return (
    <div
      data-testid="vram-breakdown-bar"
      className="absolute inset-[2px] flex gap-[3px]"
    >
      <div className="h-full rounded-[3px]" style={{ flex: weightsBytes, backgroundColor: weightsColor }} />
      {contextBytes > 0 && (
        <div className="h-full rounded-[3px]" style={{ flex: contextBytes, backgroundColor: contextColor }} />
      )}
    </div>
  );
}

function findModelKey(modelName: string, modelKeyLookup?: Map<string, string>): string | null {
  return modelKeyLookup?.get(modelName) ?? null;
}
