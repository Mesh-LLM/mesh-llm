import { useCallback, useRef, useState, type Ref } from 'react'
import { Code2, ListEnd, MessageSquareX, Paperclip, RotateCcw, Send, Square, X } from 'lucide-react'
import { Tooltip } from '@/components/ui/tooltip'
import { cn } from '@/lib/cn'
import { resolveImageFiles } from '@/lib/attachments'

type ComposerProcessingStage = 'downloading' | 'starting' | 'processing'

/** A prompt value, or an updater evaluated against the latest draft so
 *  concurrent updates (e.g. two overlapping paste restorations) compose
 *  instead of overwriting each other. */
export type ComposerPromptUpdate = string | ((current: string) => string)

function candidatePasteImages(files: File[]): File[] {
  return files.filter((file) => file.type.startsWith('image/') || file.type === '')
}

function hasUndeterminedType(files: File[]): boolean {
  return files.some((file) => file.type === '')
}

type PasteSelection = { start: number; end: number; value: string }

async function attachDetectedPasteImages(
  files: File[],
  pastedText: string,
  selection: PasteSelection,
  onAttach: (files: File[]) => void,
  onChange: (value: ComposerPromptUpdate) => void
): Promise<void> {
  const resolved = await resolveImageFiles(files)
  const images = resolved.filter((file) => file.type.startsWith('image/'))
  if (images.length > 0) {
    onAttach(images)
    return
  }
  // No image materialized, so the suppressed default paste would have
  // inserted this text at the selection. Evaluate against the latest
  // draft so concurrent restorations compose: when the prompt is
  // untouched since the paste, put the text back exactly there,
  // replacing what the selection covered; if it was edited meanwhile,
  // append instead of applying a stale range over the edit.
  if (pastedText) {
    onChange((current) =>
      current === selection.value
        ? `${current.slice(0, selection.start)}${pastedText}${current.slice(selection.end)}`
        : `${current}${pastedText}`
    )
  }
}

type ComposerProps = {
  value: string
  onChange: (value: ComposerPromptUpdate) => void
  onSend: () => void
  onAttach?: (files: File[]) => void
  onSystemPrompt?: () => void
  attachments?: File[]
  onRemoveAttachment?: (index: number) => void
  onStop?: () => void
  onRetry?: () => void
  canRetry?: boolean
  isStreaming?: boolean
  sendMode?: 'send' | 'queue'
  errorMessage?: string
  placeholder?: string
  disabled?: boolean
  requestDisabled?: boolean
  isPreparingAttachments?: boolean
  preparingStage?: ComposerProcessingStage
  preparingAttachmentCount?: number
  textareaRef?: Ref<HTMLTextAreaElement>
  systemPromptButtonRef?: Ref<HTMLButtonElement>
  showSystemPromptButton?: boolean
}

function getPreparingLabel(stage: ComposerProcessingStage | undefined): string {
  if (stage === 'starting') return 'Starting local analyzer…'
  if (stage === 'processing') return 'Processing attachments…'
  return 'Downloading browser model…'
}

export function Composer({
  value,
  onChange,
  onSend,
  onAttach,
  onSystemPrompt,
  attachments = [],
  onRemoveAttachment,
  onStop,
  onRetry,
  canRetry = false,
  isStreaming = false,
  sendMode,
  errorMessage,
  placeholder = 'Ask me anything...',
  disabled = false,
  requestDisabled = false,
  isPreparingAttachments = false,
  preparingStage,
  preparingAttachmentCount = 0,
  textareaRef,
  systemPromptButtonRef,
  showSystemPromptButton = false
}: ComposerProps) {
  const attachmentInputRef = useRef<HTMLInputElement | null>(null)
  const [pendingTypeSniffs, setPendingTypeSniffs] = useState(0)
  const runTypeSniff = useCallback((run: () => Promise<void>) => {
    setPendingTypeSniffs((count) => count + 1)
    run()
      .catch((error) => console.warn('attachment type detection failed', error))
      .finally(() => setPendingTypeSniffs((count) => count - 1))
  }, [])
  const handleSend = useCallback(() => {
    if (!disabled && !requestDisabled && pendingTypeSniffs === 0 && (value.trim() || attachments.length > 0)) {
      onSend()
    }
  }, [attachments.length, disabled, requestDisabled, pendingTypeSniffs, value, onSend])

  const sendDisabled =
    disabled || requestDisabled || pendingTypeSniffs > 0 || (!value.trim() && attachments.length === 0)
  const retryDisabled = disabled || requestDisabled || !canRetry
  const stopDisabled = disabled || !isStreaming || !onStop
  const submitQueuesPrompt = sendMode === 'queue' || isStreaming
  const preparingLabel = getPreparingLabel(preparingStage)

  return (
    <div
      className={cn(
        'composer-shell panel-shell overflow-hidden rounded-[var(--radius-lg)] border bg-panel transition-[border-color,box-shadow]',
        isPreparingAttachments
          ? 'border-[color:color-mix(in_oklab,var(--color-accent)_58%,var(--color-border))] shadow-[0_0_0_1px_color-mix(in_oklab,var(--color-accent)_18%,transparent),0_18px_55px_color-mix(in_oklab,var(--color-accent)_10%,transparent)]'
          : 'border-border'
      )}
      data-panel-interactive="true"
    >
      {attachments.length > 0 ? (
        <div
          className="flex flex-wrap items-center gap-1.5 border-b border-border-soft px-3.5 py-2"
          data-testid="composer-attachments"
        >
          {attachments.map((file, index) => {
            const fileName = file.name || `Attachment ${index + 1}`
            return (
              <span
                key={`${index}:${fileName}`}
                className="inline-flex max-w-56 items-center gap-1.5 rounded-[var(--radius)] border border-border bg-panel-strong px-2 py-1 text-[length:var(--density-type-caption)] text-fg-muted"
              >
                <Paperclip aria-hidden={true} className="size-3 shrink-0 text-fg-faint" />
                <span className="truncate">{fileName}</span>
                <Tooltip content={`Remove ${fileName}`}>
                  <button
                    type="button"
                    className="ui-control inline-flex size-4 shrink-0 items-center justify-center rounded-[var(--radius)] border"
                    aria-label={`Remove attachment ${fileName}`}
                    disabled={disabled}
                    onClick={() => onRemoveAttachment?.(index)}
                  >
                    <X className="size-2.5" />
                  </button>
                </Tooltip>
              </span>
            )
          })}
        </div>
      ) : null}
      <label className="sr-only" htmlFor="prompt-composer">
        Prompt
      </label>
      <textarea
        ref={textareaRef}
        id="prompt-composer"
        data-testid="chat-input"
        className="block w-full resize-none bg-transparent px-4 py-3.5 text-[16px] leading-[1.5] outline-none focus-visible:outline-none placeholder:text-fg-faint sm:text-[length:var(--density-type-body-lg)]"
        disabled={disabled}
        placeholder={isPreparingAttachments ? preparingLabel : placeholder}
        value={value}
        style={{ minHeight: 88, fontFamily: 'var(--font-sans)' }}
        onChange={(event) => onChange(event.target.value)}
        onPaste={(event) => {
          if (!onAttach) return

          const candidates = candidatePasteImages(Array.from(event.clipboardData?.files ?? []))
          if (candidates.length === 0) return

          // preventDefault must run synchronously; the sniffed type of an
          // undetermined file only resolves afterwards. If nothing turns out
          // to be an image, the suppressed text payload is restored by
          // attachDetectedPasteImages.
          event.preventDefault()
          if (!hasUndeterminedType(candidates)) {
            onAttach(candidates)
            return
          }
          const pastedText = event.clipboardData?.getData('text/plain') ?? ''
          const textarea = event.currentTarget
          const selection = {
            start: textarea.selectionStart ?? textarea.value.length,
            end: textarea.selectionEnd ?? textarea.value.length,
            value: textarea.value
          }
          runTypeSniff(() => attachDetectedPasteImages(candidates, pastedText, selection, onAttach, onChange))
        }}
        onKeyDown={(event) => {
          if (event.key === 'Escape' && isStreaming && onStop) {
            event.preventDefault()
            onStop()
            return
          }

          if (event.key === 'Enter' && !event.shiftKey) {
            event.preventDefault()
            handleSend()
          }
        }}
      />
      <div className="flex items-center justify-between gap-3 border-t border-border-soft bg-panel px-3.5 py-2">
        <div className="flex shrink-0 items-center gap-1.5">
          <Tooltip content="Attach files">
            <button
              type="button"
              className="ui-control inline-flex size-7 items-center justify-center rounded-[var(--radius)] border"
              aria-label="Attach"
              disabled={disabled}
              onClick={() => attachmentInputRef.current?.click()}
            >
              <Paperclip className="size-[13px]" />
            </button>
          </Tooltip>
          <input
            ref={attachmentInputRef}
            hidden
            multiple
            type="file"
            disabled={disabled}
            onChange={(event) => {
              const files = Array.from(event.target.files ?? [])
              event.target.value = ''
              if (files.length === 0) return
              if (!hasUndeterminedType(files)) {
                onAttach?.(files)
                return
              }
              runTypeSniff(() =>
                resolveImageFiles(files).then((resolved) => {
                  onAttach?.(resolved)
                })
              )
            }}
          />
          {showSystemPromptButton ? (
            <Tooltip content="Set system prompt">
              <button
                ref={systemPromptButtonRef}
                type="button"
                className="ui-control inline-flex size-7 items-center justify-center rounded-[var(--radius)] border"
                aria-label="System prompt"
                disabled={disabled}
                onClick={onSystemPrompt}
              >
                <Code2 className="size-[13px]" />
              </button>
            </Tooltip>
          ) : null}
          <Tooltip content="Retry last response">
            <button
              type="button"
              className="ui-control inline-flex size-7 items-center justify-center rounded-[var(--radius)] border"
              aria-label="Retry last"
              disabled={retryDisabled}
              onClick={onRetry}
            >
              <RotateCcw className="size-[13px]" />
            </button>
          </Tooltip>
        </div>
        <div className="flex min-w-0 flex-1 items-center justify-end gap-3 text-[length:var(--density-type-caption)]">
          <div className="min-w-0 text-right">
            {errorMessage ? (
              <div className="space-y-2 text-left">
                <div
                  role="alert"
                  className="flex items-start gap-2.5 rounded-[var(--radius)] border border-[color:color-mix(in_oklab,var(--color-bad)_38%,var(--color-border))] bg-[color:color-mix(in_oklab,var(--color-bad)_10%,transparent)] px-3 py-2.5"
                >
                  <MessageSquareX aria-hidden={true} className="mt-0.5 size-5 shrink-0 text-bad" strokeWidth={1.5} />
                  <div className="min-w-0 space-y-1">
                    <div className="font-medium text-bad">Message failed to send</div>
                    <div className="break-words text-fg-muted">{errorMessage}</div>
                    <div className="text-fg-faint">Try selecting another model, then send the prompt again.</div>
                  </div>
                </div>
                {attachments.length > 0 ? (
                  <span className="block text-fg-faint">
                    {attachments.length} attachment{attachments.length === 1 ? '' : 's'} ready
                  </span>
                ) : null}
              </div>
            ) : isPreparingAttachments ? (
              <span className="inline-flex items-center gap-2 text-accent">
                <span className="size-1.5 animate-pulse rounded-full bg-accent" aria-hidden={true} />
                {preparingLabel}{' '}
                {preparingAttachmentCount > 0 ? (
                  <span className="text-fg-faint">
                    {preparingAttachmentCount} file{preparingAttachmentCount === 1 ? '' : 's'}
                  </span>
                ) : null}
              </span>
            ) : isStreaming ? (
              <span className="text-fg-faint">Generating response… queue another prompt or stop.</span>
            ) : submitQueuesPrompt ? (
              <span className="text-fg-faint">Chat providers are busy… queue this prompt for this conversation.</span>
            ) : (
              <div className="hidden text-fg-faint md:block">
                <kbd
                  className="rounded border border-border bg-panel-strong px-[5px] py-px font-mono text-[length:var(--density-type-label)] text-fg-dim"
                  style={{ borderBottomWidth: 2 }}
                >
                  ↵
                </kbd>{' '}
                to send ·{' '}
                <kbd
                  className="rounded border border-border bg-panel-strong px-[5px] py-px font-mono text-[length:var(--density-type-label)] text-fg-dim"
                  style={{ borderBottomWidth: 2 }}
                >
                  ⇧ ↵
                </kbd>{' '}
                for newline
              </div>
            )}
          </div>
          {isStreaming ? (
            <button
              className="ui-control-destructive inline-flex items-center gap-1.5 rounded-[var(--radius)] px-3 py-1.5 text-[length:var(--density-type-control-lg)] font-medium"
              disabled={stopDisabled}
              type="button"
              onClick={onStop}
            >
              <Square className="size-[13px]" />
              Stop
            </button>
          ) : null}
          <button
            data-testid="chat-send"
            className="ui-control-primary inline-flex items-center gap-1.5 rounded-[var(--radius)] px-3 py-1.5 text-[length:var(--density-type-control-lg)] font-medium"
            disabled={sendDisabled}
            type="button"
            onClick={handleSend}
          >
            {submitQueuesPrompt ? <ListEnd className="size-[13px]" /> : <Send className="size-[13px]" />}
            {submitQueuesPrompt ? 'Queue' : 'Send'}
          </button>
        </div>
      </div>
    </div>
  )
}
