import * as SliderPrimitive from '@radix-ui/react-slider'
import { animate } from 'animejs'
import { useCallback, useEffect, useLayoutEffect, useRef, useState } from 'react'
import { CTX_MAX, CTX_MIN, CTX_TICKS, fmtCtx, stepCtx } from '@/features/configuration/components/ctx-slider-utils'

type ExactValuePosition = 'top-left' | 'top-right' | 'bottom-left' | 'bottom-right' | 'inline-left' | 'inline-right'

type CtxSliderProps = {
  value: number
  onChange: (value: number) => void
  maxCtx: number
  minSelectableCtx?: number
  maxSelectableCtx?: number
  invalid?: boolean
  disabled?: boolean
  ariaDescribedBy?: string
  ariaLabel?: string
  controlTabIndex?: number
  exactValuePosition?: ExactValuePosition
  exactValueVisibility?: 'hidden' | 'shown'
  showHeader?: boolean
}

const THUMB_SIZE_PX = 13

function selectableMinCtx(minSelectableCtx: number | undefined) {
  if (!Number.isFinite(minSelectableCtx)) return CTX_MIN
  return Math.min(CTX_MAX, Math.max(CTX_MIN, Math.round(minSelectableCtx ?? CTX_MIN)))
}

function selectableMaxCtx(maxSelectableCtx: number | undefined, minCtx: number) {
  if (!Number.isFinite(maxSelectableCtx)) return CTX_MAX
  return Math.min(CTX_MAX, Math.max(minCtx, Math.round(maxSelectableCtx ?? CTX_MAX)))
}

function boundCtxToRange(ctx: number, minCtx: number, maxCtx: number): number {
  if (!Number.isFinite(ctx)) return minCtx
  return Math.min(maxCtx, Math.max(minCtx, ctx))
}

function normalizeCtxToRange(ctx: number, minCtx: number, maxCtx: number): number {
  return Math.round(boundCtxToRange(ctx, minCtx, maxCtx))
}

function ctxToPctWithin(ctx: number, minCtx: number, maxCtx: number): number {
  if (maxCtx <= minCtx) return 0
  const bounded = boundCtxToRange(ctx, minCtx, maxCtx)
  return ((Math.log2(bounded) - Math.log2(minCtx)) / (Math.log2(maxCtx) - Math.log2(minCtx))) * 100
}

function pctToCtxWithin(pct: number, minCtx: number, maxCtx: number): number {
  const boundedPct = Math.min(100, Math.max(0, pct))
  return 2 ** (Math.log2(minCtx) + (boundedPct / 100) * (Math.log2(maxCtx) - Math.log2(minCtx)))
}

function exactValueAlignment(position: ExactValuePosition): 'left' | 'right' {
  return position.endsWith('left') ? 'left' : 'right'
}

function exactValuePlacement(position: ExactValuePosition): 'top' | 'bottom' | 'inline' {
  if (position.startsWith('top')) return 'top'
  if (position.startsWith('inline')) return 'inline'
  return 'bottom'
}

export function CtxSlider({
  value,
  onChange,
  maxCtx,
  minSelectableCtx,
  maxSelectableCtx,
  invalid = false,
  disabled = false,
  ariaDescribedBy,
  ariaLabel = 'Context',
  controlTabIndex,
  exactValuePosition = 'bottom-right',
  exactValueVisibility = 'hidden',
  showHeader = true
}: CtxSliderProps) {
  const trackAlertRef = useRef<HTMLSpanElement>(null)
  const fillAlertRef = useRef<HTMLSpanElement>(null)
  const knobAlertRef = useRef<HTMLSpanElement>(null)
  const alertAnimationsRef = useRef<Array<ReturnType<typeof animate>>>([])
  const pointerInteractingRef = useRef(false)
  const latestRawCtx = useRef(value)
  const latestEmittedCtx = useRef(value)
  const [dragging, setDragging] = useState(false)
  const [draftCtx, setDraftCtx] = useState(value)
  const [hoveredTick, setHoveredTick] = useState<number | null>(null)
  const minSelectable = selectableMinCtx(minSelectableCtx)
  const maxSelectable = selectableMaxCtx(maxSelectableCtx, minSelectable)
  const ticks = CTX_TICKS.filter((tick) => tick >= minSelectable && tick <= maxSelectable)

  useEffect(
    () => () => {
      alertAnimationsRef.current.forEach((animation) => {
        animation.pause()
      })
    },
    []
  )

  useEffect(() => {
    if (dragging) return
    latestRawCtx.current = value
    latestEmittedCtx.current = value
  }, [dragging, value])

  const commitCtx = useCallback(
    (nextCtx: number) => {
      if (disabled) return
      if (!Number.isFinite(nextCtx)) return
      const raw = boundCtxToRange(nextCtx, minSelectable, maxSelectable)
      const next = normalizeCtxToRange(raw, minSelectable, maxSelectable)
      latestRawCtx.current = raw
      setDraftCtx(raw)
      if (next === latestEmittedCtx.current) return
      latestEmittedCtx.current = next
      onChange(next)
    },
    [disabled, maxSelectable, minSelectable, onChange]
  )

  const commitPct = useCallback(
    (nextPct: number) => {
      if (!Number.isFinite(nextPct)) return
      commitCtx(pctToCtxWithin(nextPct, minSelectable, maxSelectable))
    },
    [commitCtx, maxSelectable, minSelectable]
  )

  const handleValueChange = useCallback(
    (nextValues: number[]) => {
      const nextPct = nextValues[0]
      if (!Number.isFinite(nextPct)) return

      if (pointerInteractingRef.current) {
        commitPct(nextPct)
        return
      }

      if (nextPct <= 0) {
        commitCtx(minSelectable)
        return
      }

      if (nextPct >= 100) {
        commitCtx(maxSelectable)
        return
      }

      const currentPct = ctxToPctWithin(latestEmittedCtx.current, minSelectable, maxSelectable)
      if (nextPct === currentPct) return

      commitCtx(
        boundCtxToRange(stepCtx(latestEmittedCtx.current, nextPct > currentPct ? 1 : -1), minSelectable, maxSelectable)
      )
    },
    [commitCtx, commitPct, maxSelectable, minSelectable]
  )

  const handleValueCommit = useCallback(
    (nextValues: number[]) => {
      const nextPct = nextValues[0]
      if (!Number.isFinite(nextPct)) return
      if (!pointerInteractingRef.current) return
      commitPct(nextPct)
      pointerInteractingRef.current = false
      setDragging(false)
    },
    [commitPct]
  )

  const displayCtx = dragging ? draftCtx : value

  const valuePct = ctxToPctWithin(displayCtx, minSelectable, maxSelectable)
  // Radix nudges thumbs inward near the bounds; offset the visual knob back onto the log-scale value.
  const thumbInBoundsOffsetPx = (THUMB_SIZE_PX / 2) * (1 - valuePct / 50)
  const thumbTransform = `translateX(${-thumbInBoundsOffsetPx}px)${dragging ? ' scale(1.1)' : ''}`
  const dangerStartPct = ctxToPctWithin(maxCtx, minSelectable, maxSelectable)
  const showDanger = maxCtx < maxSelectable
  const overAllocated = invalid || displayCtx > maxCtx
  const valueText =
    displayCtx > maxCtx
      ? `${fmtCtx(displayCtx)} context exceeds ${fmtCtx(maxCtx)} safe limit`
      : `${fmtCtx(displayCtx)} context`
  const exactValue = normalizeCtxToRange(displayCtx, minSelectable, maxSelectable).toLocaleString()
  const showExactValue = exactValueVisibility === 'shown'
  const exactPlacement = exactValuePlacement(exactValuePosition)
  const exactAlignment = exactValueAlignment(exactValuePosition)
  const exactValueBadge = showExactValue ? (
    <span className="rounded-[3px] border border-border-soft bg-panel-strong px-1.5 py-0.5 text-[length:var(--density-type-micro)] text-fg-faint">
      <span className="font-mono text-fg">{exactValue}</span> tokens
    </span>
  ) : null

  const exactValueBlock = exactValueBadge ? (
    <div className={`flex ${exactAlignment === 'left' ? 'justify-start' : 'justify-end'}`}>{exactValueBadge}</div>
  ) : null

  useLayoutEffect(() => {
    const alertTargets = [trackAlertRef.current, fillAlertRef.current, knobAlertRef.current].filter(
      (target): target is HTMLElement => target !== null
    )
    if (alertTargets.length === 0) return

    alertAnimationsRef.current.forEach((animation) => {
      animation.pause()
    })
    const reduceMotion =
      typeof window !== 'undefined' &&
      typeof window.matchMedia === 'function' &&
      window.matchMedia('(prefers-reduced-motion: reduce)').matches

    alertAnimationsRef.current = [
      animate(alertTargets, {
        opacity: overAllocated ? 1 : 0,
        duration: reduceMotion ? 0 : 180,
        ease: 'out(4)'
      })
    ]
  }, [overAllocated])

  const sliderRoot = (
    <SliderPrimitive.Root
      aria-describedby={ariaDescribedBy}
      aria-invalid={overAllocated}
      className={`relative flex h-6 w-full touch-none select-none items-center rounded-[var(--radius)] outline-none transition-[box-shadow] duration-150 focus-visible:shadow-[inset_0_0_0_2px_var(--color-accent)] focus-visible:[&_[data-ctx-slider-track]]:border-accent ${disabled ? 'cursor-not-allowed opacity-60' : 'cursor-pointer'}`}
      disabled={disabled}
      max={100}
      min={0}
      onPointerDown={() => {
        if (disabled) return
        pointerInteractingRef.current = true
        setDragging(true)
      }}
      onPointerCancel={() => {
        pointerInteractingRef.current = false
        setDragging(false)
      }}
      onValueChange={handleValueChange}
      onValueCommit={handleValueCommit}
      step={0.001}
      value={[valuePct]}
    >
      <SliderPrimitive.Track
        data-ctx-slider-track
        className="relative h-full grow overflow-hidden rounded-[var(--radius)] border border-border-soft bg-muted transition-[border-color] duration-150"
      >
        <SliderPrimitive.Range className="absolute inset-y-0 left-0 bg-accent" />
        <span
          ref={fillAlertRef}
          aria-hidden="true"
          className="pointer-events-none absolute inset-y-0 left-0 bg-bad opacity-0"
          style={{ width: `${valuePct}%` }}
        />
        {showDanger ? (
          <span
            aria-hidden="true"
            className="absolute inset-y-0 right-0 rounded-r-[var(--radius)] border-l border-dashed border-bad/80 opacity-75"
            style={{
              left: `${dangerStartPct}%`,
              backgroundImage:
                'repeating-linear-gradient(135deg, color-mix(in oklch, var(--color-bad) 42%, transparent) 0 3px, transparent 3px 7px)'
            }}
          />
        ) : null}
        {ticks.map((tick) => (
          <span
            aria-hidden="true"
            className="absolute top-1/2 h-3 -translate-x-1/2 -translate-y-1/2 border-l border-background/60"
            key={tick}
            style={{ left: `${ctxToPctWithin(tick, minSelectable, maxSelectable)}%` }}
          />
        ))}
      </SliderPrimitive.Track>
      <SliderPrimitive.Thumb
        aria-invalid={overAllocated}
        aria-label={ariaLabel}
        aria-valuemax={maxSelectable}
        aria-valuemin={minSelectable}
        aria-valuenow={normalizeCtxToRange(displayCtx, minSelectable, maxSelectable)}
        aria-valuetext={valueText}
        className="relative z-10 block size-[13px] overflow-hidden rounded-full border border-panel bg-accent shadow-[var(--shadow-slider-thumb)] outline-none transition-transform duration-150"
        data-ctx-slider-thumb
        style={{ transform: thumbTransform }}
        tabIndex={disabled ? -1 : controlTabIndex}
      >
        <span ref={knobAlertRef} aria-hidden="true" className="absolute inset-0 rounded-full bg-bad opacity-0" />
      </SliderPrimitive.Thumb>
      <span
        ref={trackAlertRef}
        aria-hidden="true"
        className="pointer-events-none absolute inset-0 rounded-[var(--radius)] border border-bad opacity-0 shadow-[var(--shadow-slider-alert)]"
      />
    </SliderPrimitive.Root>
  )

  return (
    <div className="select-none">
      {showHeader ? (
        <div className="mb-1.5 flex items-center justify-between gap-3">
          <span className="text-[length:var(--density-type-caption)] font-medium text-fg-dim">Context</span>
          <span className={`font-mono text-[length:var(--density-type-caption)] ${invalid ? 'text-bad' : 'text-fg'}`}>
            {fmtCtx(displayCtx)} ctx
          </span>
        </div>
      ) : null}
      {exactPlacement === 'top' && exactValueBlock ? <div className="mb-1">{exactValueBlock}</div> : null}
      {exactPlacement === 'inline' && exactValueBadge ? (
        <div className="flex items-center gap-2">
          {exactAlignment === 'left' ? exactValueBadge : null}
          <div className="min-w-0 flex-1">{sliderRoot}</div>
          {exactAlignment === 'right' ? exactValueBadge : null}
        </div>
      ) : (
        sliderRoot
      )}
      {exactPlacement === 'bottom' && exactValueBlock ? <div className="mt-1">{exactValueBlock}</div> : null}
      <div className="relative mt-1 h-4">
        {ticks.map((tick) => {
          const active = hoveredTick === tick || normalizeCtxToRange(displayCtx, minSelectable, maxSelectable) === tick
          const unsafe = tick > maxCtx
          return (
            <button
              className={`absolute -translate-x-1/2 rounded-[3px] px-1 py-0.5 font-mono text-[length:var(--density-type-micro)] transition-[background,color] duration-150 disabled:cursor-not-allowed disabled:opacity-50 ${active ? '' : unsafe ? 'text-bad' : 'text-fg-faint hover:bg-muted hover:text-fg'}`}
              disabled={disabled}
              key={tick}
              onClick={() => commitCtx(tick)}
              onMouseEnter={() => setHoveredTick(tick)}
              onMouseLeave={() => setHoveredTick(null)}
              style={{
                left: `${ctxToPctWithin(tick, minSelectable, maxSelectable)}%`,
                ...(active ? { background: 'var(--color-accent)', color: 'var(--color-accent-ink)' } : undefined)
              }}
              tabIndex={disabled ? -1 : controlTabIndex}
              title={`Set context to ${fmtCtx(tick)}`}
              type="button"
            >
              {fmtCtx(tick)}
            </button>
          )
        })}
      </div>
    </div>
  )
}
