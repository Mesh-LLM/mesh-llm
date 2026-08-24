import { useCallback, type Dispatch, type MutableRefObject, type SetStateAction } from 'react'
import {
  createObjectUrl,
  getSubmittedAttachmentKind,
  getSubmittedAttachmentLabel,
  revokeObjectUrl
} from '@/features/chat/pages/chat-page-attachments'
import type { SubmittedAttachmentKind, SubmittedAttachmentPreview } from '@/features/chat/pages/chat-page-attachments'

type SubmittedAttachmentsByMessageId = Record<string, SubmittedAttachmentPreview[]>

export type ChatPageSubmittedAttachmentsState = {
  submittedAttachmentsByMessageId: SubmittedAttachmentsByMessageId
  setSubmittedAttachmentsByMessageId: Dispatch<SetStateAction<SubmittedAttachmentsByMessageId>>
  selectedAttachmentPreview: SubmittedAttachmentPreview | null
  setSelectedAttachmentPreview: Dispatch<SetStateAction<SubmittedAttachmentPreview | null>>
  submittedAttachmentUrlsRef: MutableRefObject<Set<string>>
  createSubmittedAttachmentPreviews: (
    attachments: File[],
    conversationId: string,
    messageId: string
  ) => SubmittedAttachmentPreview[]
  removeSubmittedAttachmentPreviewsForConversation: (conversationId: string) => void
  removeSubmittedAttachmentPreviewsForMessage: (messageId: string) => void
}

type ChatPageSubmittedAttachmentsInput = Pick<
  ChatPageSubmittedAttachmentsState,
  'setSubmittedAttachmentsByMessageId' | 'setSelectedAttachmentPreview' | 'submittedAttachmentUrlsRef'
>

type ChatPageSubmittedAttachmentsActions = Pick<
  ChatPageSubmittedAttachmentsState,
  | 'createSubmittedAttachmentPreviews'
  | 'removeSubmittedAttachmentPreviewsForConversation'
  | 'removeSubmittedAttachmentPreviewsForMessage'
>

export function useChatPageSubmittedAttachments({
  setSubmittedAttachmentsByMessageId,
  setSelectedAttachmentPreview,
  submittedAttachmentUrlsRef
}: ChatPageSubmittedAttachmentsInput): ChatPageSubmittedAttachmentsActions {
  const revokeSubmittedAttachmentPreviews = useCallback(
    (previews: SubmittedAttachmentPreview[]) => {
      for (const preview of previews) {
        revokeObjectUrl(preview.objectUrl)
        submittedAttachmentUrlsRef.current.delete(preview.objectUrl)
      }
    },
    [submittedAttachmentUrlsRef]
  )

  const createSubmittedAttachmentPreviews = useCallback(
    (attachments: File[], conversationId: string, messageId: string): SubmittedAttachmentPreview[] => {
      const counters: Record<SubmittedAttachmentKind, number> = { image: 0, pdf: 0, audio: 0, file: 0 }

      return attachments.map((attachment, index) => {
        const kind = getSubmittedAttachmentKind(attachment)
        counters[kind] += 1
        const objectUrl = createObjectUrl(attachment)
        if (objectUrl) submittedAttachmentUrlsRef.current.add(objectUrl)

        return {
          id: `${attachment.name}-${attachment.lastModified}-${index}`,
          conversationId,
          messageId,
          label: getSubmittedAttachmentLabel(kind, counters[kind]),
          kind,
          fileName: attachment.name || getSubmittedAttachmentLabel(kind, counters[kind]),
          mimeType: attachment.type,
          objectUrl
        }
      })
    },
    [submittedAttachmentUrlsRef]
  )

  const removeSubmittedAttachmentPreviewsForConversation = useCallback(
    (conversationId: string) => {
      setSubmittedAttachmentsByMessageId((current) => {
        let changed = false
        const next = { ...current }

        for (const [messageId, previews] of Object.entries(current)) {
          const removedPreviews = previews.filter((preview) => preview.conversationId === conversationId)
          if (removedPreviews.length === 0) continue

          const keptPreviews = previews.filter((preview) => preview.conversationId !== conversationId)
          revokeSubmittedAttachmentPreviews(removedPreviews)
          if (keptPreviews.length > 0) {
            next[messageId] = keptPreviews
          } else {
            delete next[messageId]
          }
          changed = true
        }

        return changed ? next : current
      })
      setSelectedAttachmentPreview((current) => (current?.conversationId === conversationId ? null : current))
    },
    [revokeSubmittedAttachmentPreviews, setSubmittedAttachmentsByMessageId, setSelectedAttachmentPreview]
  )

  const removeSubmittedAttachmentPreviewsForMessage = useCallback(
    (messageId: string) => {
      setSubmittedAttachmentsByMessageId((current) => {
        const previews = current[messageId]
        if (!previews) return current

        revokeSubmittedAttachmentPreviews(previews)
        const next = { ...current }
        delete next[messageId]
        return next
      })
      setSelectedAttachmentPreview((current) => (current?.messageId === messageId ? null : current))
    },
    [revokeSubmittedAttachmentPreviews, setSubmittedAttachmentsByMessageId, setSelectedAttachmentPreview]
  )

  return {
    createSubmittedAttachmentPreviews,
    removeSubmittedAttachmentPreviewsForConversation,
    removeSubmittedAttachmentPreviewsForMessage
  }
}
