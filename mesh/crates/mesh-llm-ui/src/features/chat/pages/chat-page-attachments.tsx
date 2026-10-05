/* eslint-disable react-refresh/only-export-components */
import * as DialogPrimitive from '@radix-ui/react-dialog'
import { Check, Download, FileIcon, FileImage, FileText, Loader2, Music, Play, ScanText, X } from 'lucide-react'
import type { AttachmentProcessingStage } from '@/features/chat/api/legacy-attachments'
import type { MessageAttachmentAction } from '@/features/chat/components/MessageRow'
import { cn } from '@/lib/utils'

export type AttachmentProcessingStatus = {
  conversationId: string
  stage: AttachmentProcessingStage
  attachmentCount: number
  prompt: string
  usesBrowserAnalyzer: boolean
  browserAnalyzerReady: boolean
}

export type SubmittedAttachmentKind = MessageAttachmentAction['kind']

export type SubmittedAttachmentPreview = {
  id: string
  conversationId: string
  messageId: string
  label: string
  kind: SubmittedAttachmentKind
  fileName: string
  mimeType: string
  objectUrl: string
}

export const ATTACHMENT_PROCESSING_ORDER: Record<AttachmentProcessingStage, number> = {
  downloading: 0,
  starting: 1,
  processing: 2
}

const ATTACHMENT_PROCESSING_STEPS: Array<{
  stage: AttachmentProcessingStage
  title: string
  description: string
}> = [
  {
    stage: 'downloading',
    title: 'Downloading',
    description: 'Fetching the browser analyzer and attachment assets.'
  },
  {
    stage: 'starting',
    title: 'Starting',
    description: 'Warming the local vision and document pipeline.'
  },
  {
    stage: 'processing',
    title: 'Processing',
    description: 'Reading the attachment before the prompt is sent.'
  }
]

export function usesBrowserAnalyzerForAttachment(file: File): boolean {
  const mimeType = file.type.toLowerCase()
  const fileName = file.name.toLowerCase()
  return mimeType.startsWith('image/') || mimeType === 'application/pdf' || fileName.endsWith('.pdf')
}

function getAttachmentProcessingStepCopy(
  step: (typeof ATTACHMENT_PROCESSING_STEPS)[number],
  status: AttachmentProcessingStatus
) {
  if (!status.usesBrowserAnalyzer) {
    if (step.stage === 'downloading') {
      return { title: 'Reading', description: 'Loading the attachment bytes in this browser.' }
    }
    if (step.stage === 'starting') {
      return { title: 'Preparing', description: 'Packaging the attachment for the request.' }
    }
  }

  if (status.browserAnalyzerReady) {
    if (step.stage === 'downloading') {
      return { title: 'Cached', description: 'Reusing the browser analyzer already loaded in this tab.' }
    }
    if (step.stage === 'starting') {
      return { title: 'Ready', description: 'The local vision and document pipeline is already warm.' }
    }
  }

  return { title: step.title, description: step.description }
}

export function getSubmittedAttachmentKind(file: File): SubmittedAttachmentKind {
  const mimeType = file.type.toLowerCase()
  const fileName = file.name.toLowerCase()
  if (mimeType.startsWith('image/')) return 'image'
  if (mimeType === 'application/pdf' || fileName.endsWith('.pdf')) return 'pdf'
  if (mimeType.startsWith('audio/')) return 'audio'
  return 'file'
}

export function getSubmittedAttachmentLabel(kind: SubmittedAttachmentKind, ordinal: number): string {
  if (kind === 'image') return `Image ${ordinal}`
  if (kind === 'pdf') return `PDF ${ordinal}`
  if (kind === 'audio') return `Audio ${ordinal}`
  return `File ${ordinal}`
}

export function createObjectUrl(file: File): string {
  if (typeof URL === 'undefined' || typeof URL.createObjectURL !== 'function') return ''
  return URL.createObjectURL(file)
}

export function revokeObjectUrl(objectUrl: string) {
  if (!objectUrl || typeof URL === 'undefined' || typeof URL.revokeObjectURL !== 'function') return
  URL.revokeObjectURL(objectUrl)
}

function getAttachmentProcessingHeadline(status: AttachmentProcessingStatus): string {
  if (status.stage === 'starting') return 'Starting local analyzer'
  if (status.stage === 'processing') return 'Processing attachment content'
  return 'Downloading browser model'
}

export function AttachmentProcessingPanel({ status }: { status: AttachmentProcessingStatus }) {
  const activeIndex = ATTACHMENT_PROCESSING_ORDER[status.stage]
  const prompt = status.prompt.trim()

  return (
    <section
      aria-live="polite"
      aria-label="Attachment preparation status"
      className="mx-auto my-10 flex w-full max-w-[34rem] flex-col items-center text-center"
    >
      <div className="relative w-full overflow-hidden rounded-[calc(var(--radius-lg)+6px)] border border-[color:color-mix(in_oklab,var(--color-accent)_35%,var(--color-border))] bg-[color:color-mix(in_oklab,var(--color-panel)_88%,var(--color-accent)_12%)] p-5 shadow-[0_22px_70px_color-mix(in_oklab,var(--color-accent)_10%,transparent)]">
        <div className="absolute left-1/2 top-0 h-px w-2/3 -translate-x-1/2 bg-[color:color-mix(in_oklab,var(--color-accent)_48%,transparent)]" />
        <div className="mx-auto mb-4 flex size-12 items-center justify-center rounded-full border border-[color:color-mix(in_oklab,var(--color-accent)_40%,var(--color-border))] bg-panel-strong text-accent">
          <Loader2 className="size-5 animate-spin" aria-hidden={true} strokeWidth={1.7} />
        </div>
        <div className="space-y-1">
          <p className="text-[length:var(--density-type-label)] font-semibold uppercase tracking-[0.18em] text-accent">
            Preparing attachments
          </p>
          <h2 className="text-[length:var(--density-type-title)] font-semibold text-foreground">
            {getAttachmentProcessingHeadline(status)}
          </h2>
          <p className="mx-auto max-w-[26rem] text-[length:var(--density-type-body)] leading-6 text-fg-muted">
            {status.attachmentCount} file{status.attachmentCount === 1 ? '' : 's'} will be converted locally, then your
            prompt will be sent to the model.
          </p>
        </div>
        {prompt ? (
          <div className="mx-auto mt-4 max-w-[28rem] rounded-[var(--radius)] border border-border-soft bg-panel px-3 py-2 text-left text-[length:var(--density-type-caption)] text-fg-muted">
            <span className="font-medium text-fg">Prompt waiting:</span> {prompt}
          </div>
        ) : null}
        <ol className="mt-5 grid gap-2 text-left sm:grid-cols-3">
          {ATTACHMENT_PROCESSING_STEPS.map((step, index) => {
            const complete = index < activeIndex
            const active = index === activeIndex
            const Icon = step.stage === 'downloading' ? Download : step.stage === 'starting' ? Play : ScanText
            const copy = getAttachmentProcessingStepCopy(step, status)
            return (
              <li
                key={step.stage}
                className={cn(
                  'rounded-[var(--radius)] border px-3 py-3 transition-colors',
                  active
                    ? 'border-[color:color-mix(in_oklab,var(--color-accent)_35%,var(--color-border-soft))] bg-[color:color-mix(in_oklab,var(--color-accent)_6%,var(--color-panel))]'
                    : 'border-border-soft bg-panel'
                )}
                aria-current={active ? 'step' : undefined}
              >
                <div className="mb-2 flex items-center gap-2">
                  <span
                    className={cn(
                      'inline-flex size-6 items-center justify-center rounded-full border text-[length:var(--density-type-label)] transition-colors',
                      complete
                        ? 'border-accent bg-accent text-panel'
                        : active
                          ? 'border-accent bg-[color:color-mix(in_oklab,var(--color-accent)_18%,transparent)] text-accent'
                          : 'border-border text-fg-faint'
                    )}
                  >
                    {complete ? (
                      <Check className="size-3.5" aria-hidden={true} />
                    ) : active ? (
                      <Loader2 className="size-3.5 animate-spin" aria-hidden={true} />
                    ) : (
                      <Icon className="size-3.5" aria-hidden={true} />
                    )}
                  </span>
                  <span
                    className={cn(
                      'text-[length:var(--density-type-caption)]',
                      active ? 'font-semibold text-foreground' : 'font-medium text-fg-muted'
                    )}
                  >
                    {copy.title}
                  </span>
                </div>
                <p className="text-[length:var(--density-type-caption)] leading-5 text-fg-faint">{copy.description}</p>
              </li>
            )
          })}
        </ol>
      </div>
    </section>
  )
}

function AttachmentPreviewIcon({ kind }: { kind: SubmittedAttachmentKind }) {
  if (kind === 'image') return <FileImage className="size-4" aria-hidden={true} />
  if (kind === 'pdf') return <FileText className="size-4" aria-hidden={true} />
  if (kind === 'audio') return <Music className="size-4" aria-hidden={true} />
  return <FileIcon className="size-4" aria-hidden={true} />
}

function AttachmentPreviewBody({ attachment }: { attachment: SubmittedAttachmentPreview }) {
  if (!attachment.objectUrl) {
    return (
      <div className="grid min-h-[18rem] place-items-center rounded-[var(--radius)] border border-border-soft bg-panel-strong px-6 text-center">
        <div className="max-w-[26rem] space-y-2">
          <AttachmentPreviewIcon kind={attachment.kind} />
          <p className="text-[length:var(--density-type-body)] font-medium text-fg">Preview unavailable</p>
          <p className="text-[length:var(--density-type-control)] leading-6 text-fg-muted">
            The file was submitted with this prompt, but this browser cannot create a local preview URL for it.
          </p>
        </div>
      </div>
    )
  }

  if (attachment.kind === 'image') {
    return (
      <div className="grid max-h-[min(72vh,760px)] min-h-[18rem] place-items-center overflow-auto rounded-[var(--radius)] border border-border-soft bg-panel-strong p-3">
        <img
          src={attachment.objectUrl}
          alt={attachment.fileName}
          className="max-h-[68vh] max-w-full rounded-[calc(var(--radius)-2px)] object-contain"
        />
      </div>
    )
  }

  if (attachment.kind === 'pdf') {
    return (
      <iframe
        title={`Preview ${attachment.fileName}`}
        src={attachment.objectUrl}
        className="h-[min(72vh,760px)] w-full rounded-[var(--radius)] border border-border-soft bg-panel-strong"
      />
    )
  }

  if (attachment.kind === 'audio') {
    return (
      <div className="grid min-h-[16rem] place-items-center rounded-[var(--radius)] border border-border-soft bg-panel-strong px-6">
        <audio controls src={attachment.objectUrl} className="w-full max-w-[32rem]">
          <track kind="captions" />
        </audio>
      </div>
    )
  }

  return (
    <div className="grid min-h-[18rem] place-items-center rounded-[var(--radius)] border border-border-soft bg-panel-strong px-6 text-center">
      <div className="max-w-[26rem] space-y-2">
        <AttachmentPreviewIcon kind={attachment.kind} />
        <p className="text-[length:var(--density-type-body)] font-medium text-fg">{attachment.fileName}</p>
        <p className="text-[length:var(--density-type-control)] leading-6 text-fg-muted">
          This attachment was sent with the prompt. Inline preview is available for images, PDFs, and audio files.
        </p>
      </div>
    </div>
  )
}

export function AttachmentPreviewDialog({
  attachment,
  onOpenChange
}: {
  attachment: SubmittedAttachmentPreview | null
  onOpenChange: (open: boolean) => void
}) {
  const open = attachment !== null

  return (
    <DialogPrimitive.Root open={open} onOpenChange={onOpenChange}>
      <DialogPrimitive.Portal>
        <DialogPrimitive.Overlay className="surface-scrim fixed inset-0 z-50 data-[state=closed]:animate-out data-[state=open]:animate-in data-[state=closed]:fade-out-0 data-[state=open]:fade-in-0" />
        <DialogPrimitive.Content className="shadow-surface-modal fixed left-1/2 top-1/2 z-50 flex max-h-[calc(100vh-2rem)] w-[min(920px,calc(100vw-2rem))] -translate-x-1/2 -translate-y-1/2 flex-col overflow-hidden rounded-[var(--radius-lg)] border border-border bg-panel text-foreground outline-none data-[state=closed]:animate-out data-[state=open]:animate-in data-[state=closed]:fade-out-0 data-[state=open]:fade-in-0 data-[state=closed]:zoom-out-95 data-[state=open]:zoom-in-95">
          {attachment ? (
            <>
              <div className="flex items-start justify-between gap-4 border-b border-border-soft px-5 py-4">
                <div className="min-w-0">
                  <DialogPrimitive.Title className="flex min-w-0 items-center gap-2 text-[length:var(--density-type-headline)] font-semibold leading-5 tracking-[-0.02em] text-fg">
                    <span className="grid size-8 shrink-0 place-items-center rounded-[var(--radius)] border border-[color:color-mix(in_oklab,var(--color-accent)_34%,var(--color-border))] bg-[color:color-mix(in_oklab,var(--color-accent)_12%,var(--color-panel))] text-accent">
                      <AttachmentPreviewIcon kind={attachment.kind} />
                    </span>
                    <span className="truncate">{attachment.fileName}</span>
                  </DialogPrimitive.Title>
                  <DialogPrimitive.Description className="mt-1.5 text-[length:var(--density-type-caption)] text-fg-faint">
                    {attachment.label} {attachment.mimeType ? `· ${attachment.mimeType}` : ''}
                  </DialogPrimitive.Description>
                </div>
                <DialogPrimitive.Close asChild>
                  <button
                    type="button"
                    className="ui-control inline-flex size-8 shrink-0 items-center justify-center rounded-[var(--radius)] border text-fg-muted outline-none transition-[background,color,box-shadow,transform] hover:text-fg focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
                    aria-label="Close attachment preview"
                  >
                    <X className="size-4" aria-hidden={true} />
                  </button>
                </DialogPrimitive.Close>
              </div>
              <div className="min-h-0 overflow-auto p-4">
                <AttachmentPreviewBody attachment={attachment} />
              </div>
            </>
          ) : null}
        </DialogPrimitive.Content>
      </DialogPrimitive.Portal>
    </DialogPrimitive.Root>
  )
}
/* eslint-enable react-refresh/only-export-components */
