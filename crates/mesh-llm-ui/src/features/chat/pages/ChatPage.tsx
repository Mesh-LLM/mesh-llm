import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { Cpu, HardDrive } from 'lucide-react'
import { LiveDataUnavailableOverlay } from '@/components/ui/LiveDataUnavailableOverlay'
import { DestructiveActionDialog } from '@/components/ui/DestructiveActionDialog'
import { TextInputDialog } from '@/components/ui/TextInputDialog'
import { ChatLiveLoadingGhost } from '@/features/chat/components/ChatLiveLoadingGhost'
import { ChatSidebar } from '@/features/chat/components/ChatSidebar'
import { Composer } from '@/features/chat/components/Composer'
import { ModelSelect } from '@/features/chat/components/ModelSelect'
import { TransparencyPane } from '@/features/chat/components/transparency/TransparencyPane'
import { uiMessagesToThreadMessages } from '@/features/chat/api/use-chat-messages'
import { ChatSessionProvider } from '@/features/chat/api/chat-session'
import { createChatDraftConversationId } from '@/features/chat/api/chat-session-ids'
import { useOptionalChatSession, useChatSession } from '@/features/chat/api/chat-session-hooks'
import { buildComposerMessageContent } from '@/features/chat/api/legacy-attachments'
import {
  describeImageForPrompt,
  describeScannedPdf,
  extractPdfTextFromFile,
  isBrowserVisionModelLoaded
} from '@/features/chat/api/attachment-preprocessing'
import { useModelsQuery } from '@/features/network/api/use-models-query'
import { useStatusQuery } from '@/features/network/api/use-status-query'
import { adaptModelsToSummary } from '@/features/network/api/models-adapter'
import { useDataMode } from '@/lib/data-mode'
import { useBooleanFeatureFlag } from '@/lib/feature-flags'
import { CHAT_HARNESS } from '@/features/app-tabs/data'
import { statusBackedChatModels } from '@/features/chat/lib/live-chat-models'
import type {
  ChatActionMetric,
  ChatHarnessData,
  Conversation,
  ModelSelectOption,
  TransparencyMessage
} from '@/features/app-tabs/types'
import {
  AttachmentPreviewDialog,
  type AttachmentProcessingStatus,
  ATTACHMENT_PROCESSING_ORDER,
  createObjectUrl,
  getSubmittedAttachmentKind,
  getSubmittedAttachmentLabel,
  revokeObjectUrl,
  type SubmittedAttachmentKind,
  type SubmittedAttachmentPreview,
  usesBrowserAnalyzerForAttachment
} from '@/features/chat/pages/chat-page-attachments'
import { ChatConversationPanel } from '@/features/chat/pages/ChatConversationPanel'
import {
  AUTO_BACKEND_MODEL,
  AUTO_MODEL_OPTION,
  AUTO_MODEL_VALUE,
  isChatSelectableModel,
  modelStatusBadge
} from '@/features/chat/pages/chat-page-models'
import {
  createQueuedSubmissionId,
  createStoppedAssistantThreadMessage,
  getMessageTextContent,
  hasLastUserTurn,
  type ComposerSubmission,
  type ConversationComposerDraft,
  type DeleteConversationOptions,
  type FailedSubmission,
  type QueuedSubmission
} from '@/features/chat/pages/chat-page-submissions'

type ChatPageProps = { data?: ChatHarnessData }

function ChatMetricBadge({ metric }: { metric: ChatActionMetric }) {
  const Icon = metric.icon === 'cpu' ? Cpu : HardDrive

  return (
    <span className="hidden shrink-0 items-center gap-1.5 whitespace-nowrap rounded-full border border-border px-2.5 py-0.5 text-[length:var(--density-type-caption)] font-medium text-fg-faint md:inline-flex">
      <Icon className="size-3" /> {metric.label}
    </span>
  )
}

export function ChatPageContent({ data = CHAT_HARNESS }: ChatPageProps) {
  const { mode, setMode } = useDataMode()
  const liveMode = mode === 'live'
  const modelsQuery = useModelsQuery({ enabled: mode === 'live' })
  const statusQuery = useStatusQuery({ enabled: liveMode })
  const liveStatus = statusQuery.data
  const catalogModels = useMemo(
    () => (modelsQuery.data ? adaptModelsToSummary(modelsQuery.data.mesh_models) : undefined),
    [modelsQuery.data]
  )
  const statusModels = useMemo(() => statusBackedChatModels(liveStatus), [liveStatus])
  const liveModels = catalogModels && catalogModels.length > 0 ? catalogModels : statusModels
  const displayModels = liveMode ? liveModels : data.models
  const selectableModels = useMemo(() => displayModels.filter(isChatSelectableModel), [displayModels])
  const warmModelCount = catalogModels?.filter(isChatSelectableModel).length ?? 0
  const hasLiveReadiness = liveStatus != null || warmModelCount > 0
  const liveReadinessFetching = statusQuery.isFetching || modelsQuery.isFetching
  const showLiveError = liveMode && !hasLiveReadiness && !liveReadinessFetching && statusQuery.isError
  const showLiveLoading = liveMode && !hasLiveReadiness && liveReadinessFetching
  const canChat = !liveMode || (liveStatus?.llama_ready ?? false) || warmModelCount > 0 || statusModels.length > 0
  const transparencyTabEnabled = useBooleanFeatureFlag('chat/transparencyTab')
  const systemPromptButtonEnabled = useBooleanFeatureFlag('chat/systemPromptButton')
  const [sidebarTab, setSidebarTab] = useState<'conversations' | 'transparency'>('conversations')
  const [inspectedMessage, setInspectedMessage] = useState<TransparencyMessage | undefined>()
  const [systemPromptDialogOpen, setSystemPromptDialogOpen] = useState(false)
  const [systemPromptDraft, setSystemPromptDraft] = useState('')
  const [composerDrafts, setComposerDrafts] = useState<Record<string, ConversationComposerDraft>>({})
  const [model, setModel] = useState('')
  const modelExists = selectableModels.some((item) => item.name === model)
  // selectedModelValue is what the dropdown shows (always a value
  // present in `options`, so Radix Select can highlight it).
  // activeModelName is what we send on the wire — the "Auto" pick
  // routes through the MoA gateway via the virtual `mesh` model.
  const selectedModelValue = modelExists ? model : AUTO_MODEL_VALUE
  const activeModelName = selectedModelValue === AUTO_MODEL_VALUE ? AUTO_BACKEND_MODEL : selectedModelValue
  const [queuedSubmissions, setQueuedSubmissions] = useState<QueuedSubmission[]>([])
  const [attachmentProcessingStatus, setAttachmentProcessingStatus] = useState<AttachmentProcessingStatus | null>(null)
  const [submittedAttachmentsByMessageId, setSubmittedAttachmentsByMessageId] = useState<
    Record<string, SubmittedAttachmentPreview[]>
  >({})
  const [selectedAttachmentPreview, setSelectedAttachmentPreview] = useState<SubmittedAttachmentPreview | null>(null)
  const [failedSubmission, setFailedSubmission] = useState<FailedSubmission | null>(null)
  const [latestTurnToken, setLatestTurnToken] = useState(0)
  const queuedSubmissionsRef = useRef<QueuedSubmission[]>([])
  const queueDrainInFlightRef = useRef(false)
  const submittedAttachmentUrlsRef = useRef<Set<string>>(new Set())
  const composerTextareaRef = useRef<HTMLTextAreaElement | null>(null)
  const systemPromptButtonRef = useRef<HTMLButtonElement | null>(null)
  const [stoppedConversationIds, setStoppedConversationIds] = useState<Set<string>>(() => new Set())
  const [conversationPendingDelete, setConversationPendingDelete] = useState<Conversation | null>(null)
  const deleteDialogReturnFocusRef = useRef<HTMLElement | null>(null)

  const {
    activeConversation,
    activeConversationKey,
    activeMessages,
    chat,
    chatConversationId,
    conversations,
    createConversation,
    deleteConversation,
    draftConversationId,
    isStreaming,
    liveMessagesWithModels,
    messageCounts,
    renameConversation,
    selectConversation: persistConversationSelection,
    setDraftConversationId,
    setMessageModels,
    setSessionModel,
    setSystemPrompt,
    streamingConversationIds,
    systemPrompt,
    updateThread
  } = useChatSession()
  const pendingSendRef = useRef<{
    prompt: string
    attachments: File[]
    previousMessages: typeof chat.messages
    previousThreadMessages: ReturnType<typeof uiMessagesToThreadMessages>
    conversationId: string
    submittedModel: string
    submittedAttachmentMessageId?: string
  } | null>(null)
  const pendingRetryRef = useRef<{
    conversationId: string
    model: string
    previousAssistantMessageIds: Set<string>
  } | null>(null)
  const handledChatErrorRef = useRef<{ conversationId: string; message: string } | null>(null)
  const requestJumpToLatest = useCallback(() => {
    setLatestTurnToken((current) => current + 1)
  }, [])
  const revokeSubmittedAttachmentPreviews = useCallback((previews: SubmittedAttachmentPreview[]) => {
    for (const preview of previews) {
      revokeObjectUrl(preview.objectUrl)
      submittedAttachmentUrlsRef.current.delete(preview.objectUrl)
    }
  }, [])
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
    []
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
    [revokeSubmittedAttachmentPreviews]
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
    [revokeSubmittedAttachmentPreviews]
  )
  const displayedConversationId = activeConversationKey || chatConversationId
  const composerConversationId = displayedConversationId || draftConversationId
  const composerDraft = useMemo<ConversationComposerDraft>(() => {
    return composerDrafts[composerConversationId] ?? { prompt: '', attachments: [] }
  }, [composerConversationId, composerDrafts])
  const setComposerDraft = useCallback((conversationId: string, draft: ConversationComposerDraft) => {
    setComposerDrafts((current) => ({ ...current, [conversationId]: draft }))
  }, [])
  const updateComposerPrompt = useCallback(
    (nextPrompt: string) => {
      setComposerDrafts((current) => {
        const currentDraft = current[composerConversationId] ?? { prompt: '', attachments: [] }
        return { ...current, [composerConversationId]: { ...currentDraft, prompt: nextPrompt } }
      })
    },
    [composerConversationId]
  )
  const updateComposerAttachments = useCallback(
    (update: (current: File[]) => File[]) => {
      setComposerDrafts((current) => {
        const currentDraft = current[composerConversationId] ?? { prompt: '', attachments: [] }
        return {
          ...current,
          [composerConversationId]: { ...currentDraft, attachments: update(currentDraft.attachments) }
        }
      })
    },
    [composerConversationId]
  )
  const clearComposerDraft = useCallback(
    (conversationId: string) => setComposerDraft(conversationId, { prompt: '', attachments: [] }),
    [setComposerDraft]
  )
  const selectedConversationHasActiveLane = displayedConversationId === chatConversationId
  const composerIsStreaming = selectedConversationHasActiveLane && isStreaming
  const composerShouldQueue = isStreaming || (liveMode && !selectedConversationHasActiveLane)

  const options = useMemo<ModelSelectOption[]>(
    () => [
      AUTO_MODEL_OPTION,
      ...selectableModels.map((item) => ({
        value: item.name,
        label: item.name,
        meta: `${item.family} · ${item.context}`,
        status: modelStatusBadge(item)
      }))
    ],
    [selectableModels]
  )
  const canRetry = hasLastUserTurn(activeMessages.map((message) => ({ role: message.messageRole })))

  const inspectMessage = (message: TransparencyMessage) => {
    if (!transparencyTabEnabled) return

    setInspectedMessage(message)
    setSidebarTab('transparency')
  }
  const selectConversation = (conversation: Conversation) => {
    if (liveMode && isStreaming && liveMessagesWithModels.length > 0) {
      updateThread(chatConversationId, liveMessagesWithModels)
    }
    persistConversationSelection(conversation.id)
    setInspectedMessage(undefined)
    setSidebarTab('conversations')
  }
  const focusComposer = useCallback(() => {
    const focus = () => composerTextareaRef.current?.focus()
    if (typeof window.requestAnimationFrame === 'function') {
      window.requestAnimationFrame(focus)
      return
    }

    window.setTimeout(focus, 0)
  }, [])
  const removeQueuedSubmissionsForConversation = useCallback((conversationId: string) => {
    setQueuedSubmissions((current) => {
      const next = current.filter((submission) => submission.conversationId !== conversationId)
      queuedSubmissionsRef.current = next
      return next
    })
  }, [])
  const clearStoppedConversation = useCallback((conversationId: string) => {
    setStoppedConversationIds((current) => {
      if (!current.has(conversationId)) return current

      const next = new Set(current)
      next.delete(conversationId)
      return next
    })
  }, [])
  const requestDeleteConversation = useCallback((conversation: Conversation, options?: DeleteConversationOptions) => {
    deleteDialogReturnFocusRef.current = options?.returnFocusElement ?? null
    setConversationPendingDelete(conversation)
  }, [])
  const openSystemPromptDialog = useCallback(() => {
    setSystemPromptDraft(systemPrompt)
    setSystemPromptDialogOpen(true)
  }, [systemPrompt])
  const updateSystemPromptDialogOpen = useCallback(
    (open: boolean) => {
      if (open) setSystemPromptDraft(systemPrompt)
      setSystemPromptDialogOpen(open)
    },
    [systemPrompt]
  )
  const saveSystemPrompt = useCallback(
    (value: string) => {
      setSystemPrompt(value)
    },
    [setSystemPrompt]
  )
  const confirmDeleteSelectedConversation = useCallback(() => {
    const conversation = conversationPendingDelete
    if (!conversation) return

    const deletingSelectedConversation = conversation.id === activeConversationKey
    const deletingLiveConversation = liveMode && conversation.id === chatConversationId

    if (deletingLiveConversation) {
      if (isStreaming) chat.stop()
      chat.setMessages([])
      pendingSendRef.current = null
      pendingRetryRef.current = null
    }

    if (deletingSelectedConversation) {
      clearComposerDraft(conversation.id)
      setAttachmentProcessingStatus(null)
      setFailedSubmission(null)
      setSelectedAttachmentPreview(null)
      handledChatErrorRef.current = null
      pendingRetryRef.current = null
      setInspectedMessage(undefined)
    }

    removeQueuedSubmissionsForConversation(conversation.id)
    removeSubmittedAttachmentPreviewsForConversation(conversation.id)
    clearStoppedConversation(conversation.id)
    deleteConversation(conversation.id)
    setSidebarTab('conversations')
    focusComposer()
  }, [
    activeConversationKey,
    chat,
    chatConversationId,
    clearStoppedConversation,
    clearComposerDraft,
    conversationPendingDelete,
    deleteConversation,
    focusComposer,
    isStreaming,
    liveMode,
    removeQueuedSubmissionsForConversation,
    removeSubmittedAttachmentPreviewsForConversation
  ])
  const retryLiveData = useCallback(() => {
    void statusQuery.refetch()
  }, [statusQuery])
  const switchToTestData = useCallback(() => setMode('harness'), [setMode])

  useEffect(() => {
    setSessionModel(activeModelName)
  }, [activeModelName, setSessionModel])

  useEffect(() => {
    if (chatConversationId) focusComposer()
  }, [chatConversationId, focusComposer])

  useEffect(() => {
    const submittedAttachmentUrls = submittedAttachmentUrlsRef.current
    return () => {
      for (const objectUrl of submittedAttachmentUrls) {
        revokeObjectUrl(objectUrl)
      }
      submittedAttachmentUrls.clear()
    }
  }, [])

  useEffect(() => {
    const pendingSend = pendingSendRef.current
    if (!pendingSend) return

    const nextMessages = chat.messages.slice(pendingSend.previousMessages.length)
    const submittedUserMessage = nextMessages.find((message) => message.role === 'user')
    let submittedAttachmentMessageId = pendingSend.submittedAttachmentMessageId
    if (
      submittedUserMessage &&
      pendingSend.attachments.length > 0 &&
      !submittedAttachmentsByMessageId[submittedUserMessage.id]
    ) {
      const previews = createSubmittedAttachmentPreviews(
        pendingSend.attachments,
        pendingSend.conversationId,
        submittedUserMessage.id
      )
      setSubmittedAttachmentsByMessageId((current) => ({ ...current, [submittedUserMessage.id]: previews }))
      submittedAttachmentMessageId = submittedUserMessage.id
      pendingSendRef.current = { ...pendingSend, submittedAttachmentMessageId: submittedUserMessage.id }
    }
    const submittedMessageIds = nextMessages
      .filter((message) => message.role === 'user' || message.role === 'assistant')
      .map((message) => message.id)
    if (submittedMessageIds.length > 0) {
      setMessageModels((current) => {
        let changed = false
        const next = { ...current }
        for (const messageId of submittedMessageIds) {
          if (next[messageId] === pendingSend.submittedModel) continue
          next[messageId] = pendingSend.submittedModel
          changed = true
        }
        return changed ? next : current
      })
    }

    if (nextMessages.some((message) => message.role === 'assistant' && getMessageTextContent(message) !== '')) {
      pendingSendRef.current = null
      return
    }

    if (chat.error) {
      setComposerDraft(pendingSend.conversationId, {
        prompt: pendingSend.prompt,
        attachments: pendingSend.attachments
      })
      chat.setMessages(pendingSend.previousMessages)
      updateThread(pendingSend.conversationId, pendingSend.previousThreadMessages)
      if (submittedAttachmentMessageId) {
        removeSubmittedAttachmentPreviewsForMessage(submittedAttachmentMessageId)
      }
      handledChatErrorRef.current = { conversationId: pendingSend.conversationId, message: chat.error.message }
      setFailedSubmission({
        id: `failed-${createChatDraftConversationId()}`,
        prompt: pendingSend.prompt,
        attachments: pendingSend.attachments,
        timestamp: new Date().toISOString(),
        conversationId: pendingSend.conversationId,
        errorMessage: chat.error.message,
        model: pendingSend.submittedModel,
        includeUserRow: true
      })
      pendingSendRef.current = null
    }
  }, [
    chat,
    chat.error,
    chat.messages,
    createSubmittedAttachmentPreviews,
    removeSubmittedAttachmentPreviewsForMessage,
    setComposerDraft,
    setMessageModels,
    submittedAttachmentsByMessageId,
    updateThread
  ])

  useEffect(() => {
    const pendingRetry = pendingRetryRef.current
    if (!pendingRetry || pendingRetry.conversationId !== chatConversationId) return

    const retryAssistant = chat.messages.find(
      (message) => message.role === 'assistant' && !pendingRetry.previousAssistantMessageIds.has(message.id)
    )
    if (!retryAssistant) return

    setMessageModels((current) =>
      current[retryAssistant.id] === pendingRetry.model
        ? current
        : { ...current, [retryAssistant.id]: pendingRetry.model }
    )
    if (chat.status === 'ready' && !chat.error) pendingRetryRef.current = null
  }, [chat.error, chat.messages, chat.status, chatConversationId, setMessageModels])

  useEffect(() => {
    const pendingRetry = pendingRetryRef.current
    if (!chat.error || pendingSendRef.current || !pendingRetry) return

    const handledError = handledChatErrorRef.current
    if (handledError?.conversationId === pendingRetry.conversationId && handledError.message === chat.error.message) {
      pendingRetryRef.current = null
      return
    }

    handledChatErrorRef.current = { conversationId: pendingRetry.conversationId, message: chat.error.message }
    setFailedSubmission({
      id: `failed-${createChatDraftConversationId()}`,
      prompt: '',
      attachments: [],
      timestamp: new Date().toISOString(),
      conversationId: pendingRetry.conversationId,
      errorMessage: chat.error.message,
      model: pendingRetry.model,
      includeUserRow: false
    })
    pendingRetryRef.current = null
  }, [chat.error])

  const ensureConversation = useCallback(
    (conversationId = activeConversationKey || chatConversationId) => {
      if (activeConversationKey) return activeConversationKey

      createConversation(conversationId)
      setDraftConversationId(createChatDraftConversationId())
      return conversationId
    },
    [activeConversationKey, chatConversationId, createConversation, setDraftConversationId]
  )

  const submitPromptNow = useCallback(
    async (submission: ComposerSubmission, conversationId = activeConversationKey || chatConversationId) => {
      const promptSnapshot = submission.prompt
      const attachmentsSnapshot = [...submission.attachments]
      const ensuredConversationId = ensureConversation(conversationId)
      clearStoppedConversation(ensuredConversationId)
      setFailedSubmission(null)
      handledChatErrorRef.current = null
      pendingRetryRef.current = null
      pendingSendRef.current = {
        prompt: promptSnapshot,
        attachments: attachmentsSnapshot,
        previousMessages: chat.messages,
        previousThreadMessages: uiMessagesToThreadMessages(chat.messages),
        conversationId: ensuredConversationId,
        submittedModel: activeModelName
      }
      clearComposerDraft(ensuredConversationId)
      if (attachmentsSnapshot.length > 0) {
        const usesBrowserAnalyzer = attachmentsSnapshot.some(usesBrowserAnalyzerForAttachment)
        const browserAnalyzerReady = usesBrowserAnalyzer && isBrowserVisionModelLoaded()
        setAttachmentProcessingStatus({
          conversationId: ensuredConversationId,
          stage: browserAnalyzerReady || !usesBrowserAnalyzer ? 'processing' : 'downloading',
          attachmentCount: attachmentsSnapshot.length,
          prompt: promptSnapshot,
          usesBrowserAnalyzer,
          browserAnalyzerReady
        })
      }

      try {
        const content = await buildComposerMessageContent(submission.prompt, submission.attachments, {
          describeImage: describeImageForPrompt,
          extractPdfText: extractPdfTextFromFile,
          describeScannedPdf,
          onProcessingStage: (stage) => {
            setAttachmentProcessingStatus((current) => {
              if (!current || current.conversationId !== ensuredConversationId) return current
              if (ATTACHMENT_PROCESSING_ORDER[stage] < ATTACHMENT_PROCESSING_ORDER[current.stage]) return current
              return { ...current, stage }
            })
          }
        })
        setAttachmentProcessingStatus((current) => (current?.conversationId === ensuredConversationId ? null : current))
        await chat.sendMessage(content)
      } catch (error) {
        setAttachmentProcessingStatus((current) => (current?.conversationId === ensuredConversationId ? null : current))
        const pendingSend = pendingSendRef.current
        if (pendingSend) {
          setComposerDraft(pendingSend.conversationId, {
            prompt: promptSnapshot,
            attachments: attachmentsSnapshot
          })
          chat.setMessages(pendingSend.previousMessages)
          updateThread(pendingSend.conversationId, pendingSend.previousThreadMessages)
          pendingSendRef.current = null
        }
        const errorMessage = error instanceof Error ? error.message : String(error)
        handledChatErrorRef.current = { conversationId: ensuredConversationId, message: errorMessage }
        setFailedSubmission({
          id: `failed-${createChatDraftConversationId()}`,
          prompt: promptSnapshot,
          attachments: attachmentsSnapshot,
          timestamp: new Date().toISOString(),
          conversationId: ensuredConversationId,
          errorMessage,
          model: activeModelName,
          includeUserRow: true
        })
      }
    },
    [
      activeConversationKey,
      activeModelName,
      chat,
      chatConversationId,
      clearComposerDraft,
      clearStoppedConversation,
      ensureConversation,
      setComposerDraft,
      updateThread
    ]
  )

  const sendPrompt = useCallback(async () => {
    const nextPrompt = composerDraft.prompt.trim()
    if (!nextPrompt && composerDraft.attachments.length === 0) return

    requestJumpToLatest()
    const submission: ComposerSubmission = { prompt: composerDraft.prompt, attachments: [...composerDraft.attachments] }

    if (composerShouldQueue) {
      const queued: QueuedSubmission = {
        ...submission,
        id: createQueuedSubmissionId(),
        timestamp: new Date().toISOString(),
        conversationId: composerConversationId
      }
      setQueuedSubmissions((current) => {
        const next = [...current, queued]
        queuedSubmissionsRef.current = next
        return next
      })
      setFailedSubmission(null)
      handledChatErrorRef.current = null
      pendingRetryRef.current = null
      clearComposerDraft(composerConversationId)
      return
    }

    await submitPromptNow(submission, composerConversationId)
  }, [
    clearComposerDraft,
    composerConversationId,
    composerDraft,
    composerShouldQueue,
    requestJumpToLatest,
    submitPromptNow
  ])

  useEffect(() => {
    queuedSubmissionsRef.current = queuedSubmissions
  }, [queuedSubmissions])

  useEffect(() => {
    if (isStreaming || queueDrainInFlightRef.current || queuedSubmissions.length === 0) return

    const nextSubmission = queuedSubmissions.find((submission) => submission.conversationId === chatConversationId)
    if (!nextSubmission) return

    queueDrainInFlightRef.current = true
    setQueuedSubmissions((current) => {
      const next = current.filter((submission) => submission.id !== nextSubmission.id)
      queuedSubmissionsRef.current = next
      return next
    })
    void (async () => {
      try {
        await submitPromptNow(
          { prompt: nextSubmission.prompt, attachments: [...nextSubmission.attachments] },
          nextSubmission.conversationId
        )
      } finally {
        queueDrainInFlightRef.current = false
      }
    })()
  }, [chatConversationId, isStreaming, queuedSubmissions, submitPromptNow])

  const removeQueuedSubmission = useCallback(
    (submissionId: string) => {
      setQueuedSubmissions((current) => {
        const next = current.filter((submission) => submission.id !== submissionId)
        queuedSubmissionsRef.current = next
        return next
      })
      focusComposer()
    },
    [focusComposer]
  )

  const retryLastResponse = useCallback(async () => {
    if (!canRetry) return
    requestJumpToLatest()
    const ensuredConversationId = ensureConversation()
    clearStoppedConversation(ensuredConversationId)
    setFailedSubmission(null)
    handledChatErrorRef.current = null
    pendingRetryRef.current = {
      conversationId: ensuredConversationId,
      model: activeModelName,
      previousAssistantMessageIds: new Set(
        chat.messages.filter((message) => message.role === 'assistant').map((message) => message.id)
      )
    }
    await chat.reload()
  }, [activeModelName, canRetry, chat, clearStoppedConversation, ensureConversation, requestJumpToLatest])

  const stopStreamingResponse = useCallback(() => {
    const latestLiveMessage = liveMessagesWithModels.at(-1)
    if (liveMode && latestLiveMessage?.messageRole !== 'assistant') {
      const stoppedMessage = createStoppedAssistantThreadMessage(activeModelName)
      updateThread(chatConversationId, [...liveMessagesWithModels, stoppedMessage])
      chat.setMessages([
        ...chat.messages,
        {
          id: stoppedMessage.id,
          role: 'assistant',
          createdAt: new Date(stoppedMessage.timestamp),
          parts: [{ type: 'text', content: '' }]
        }
      ])
    } else if (liveMode && latestLiveMessage?.messageRole === 'assistant') {
      updateThread(chatConversationId, liveMessagesWithModels)
    }
    chat.stop()
    setStoppedConversationIds((current) => {
      if (current.has(chatConversationId)) return current

      const next = new Set(current)
      next.add(chatConversationId)
      return next
    })
  }, [activeModelName, chat, chatConversationId, liveMessagesWithModels, liveMode, updateThread])

  const visibleFailedSubmission =
    failedSubmission && failedSubmission.conversationId === displayedConversationId ? failedSubmission : null
  const visibleQueuedSubmissions = queuedSubmissions.filter(
    (submission) => submission.conversationId === displayedConversationId
  )
  const visibleAttachmentProcessingStatus =
    attachmentProcessingStatus && attachmentProcessingStatus.conversationId === displayedConversationId
      ? attachmentProcessingStatus
      : null
  const composerIsPreparingAttachments =
    attachmentProcessingStatus?.conversationId === composerConversationId &&
    attachmentProcessingStatus.attachmentCount > 0
  const activeConversationIsStreaming = streamingConversationIds.includes(displayedConversationId)
  const lastActiveMessage = activeMessages.at(-1)
  const lastMessageIsEmptyAssistant =
    lastActiveMessage?.messageRole === 'assistant' && lastActiveMessage.body.trim() === ''
  const lastMessageHasAssistantText =
    lastActiveMessage?.messageRole === 'assistant' && lastActiveMessage.body.trim() !== ''
  const showStreamingPlaceholder =
    activeConversationIsStreaming && !lastMessageIsEmptyAssistant && !lastMessageHasAssistantText

  const sidebar = (
    <ChatSidebar
      tab={sidebarTab}
      onTabChange={setSidebarTab}
      conversations={conversations.conversations}
      conversationGroups={conversations.conversationGroups}
      activeId={conversations.activeConversationId || activeConversation?.id}
      messageCounts={messageCounts}
      streamingConversationIds={streamingConversationIds}
      onSelectConversation={selectConversation}
      onRenameConversation={(conversation, title) => renameConversation(conversation.id, title)}
      onDeleteConversation={requestDeleteConversation}
      onNewChat={() => {
        if (liveMode && isStreaming && liveMessagesWithModels.length > 0) {
          updateThread(chatConversationId, liveMessagesWithModels)
        }
        const nextConversationId = createConversation(draftConversationId)
        setDraftConversationId(createChatDraftConversationId())
        clearComposerDraft(nextConversationId)
        setAttachmentProcessingStatus(null)
        setFailedSubmission(null)
        setSelectedAttachmentPreview(null)
        handledChatErrorRef.current = null
        pendingRetryRef.current = null
        setInspectedMessage(undefined)
        setSidebarTab('conversations')
        focusComposer()
      }}
      transparency={<TransparencyPane message={inspectedMessage} nodes={data.transparencyNodes} />}
      showTransparency={transparencyTabEnabled}
    />
  )

  const actions = (
    <>
      {data.actionMetrics.map((metric) => (
        <ChatMetricBadge key={metric.id} metric={metric} />
      ))}
      <div className="flex min-w-0 flex-1 basis-full items-center gap-2 sm:basis-auto md:flex-none">
        <span className="hidden shrink-0 whitespace-nowrap text-[length:var(--density-type-caption)] text-fg-faint md:inline">
          {data.modelLabel}
        </span>
        <ModelSelect options={options} value={selectedModelValue} onChange={setModel} />
      </div>
    </>
  )

  if (showLiveError) {
    return (
      <LiveDataUnavailableOverlay
        debugTitle="Could not reach local runtime status"
        title="Live chat is unavailable"
        debugDescription="Chat could not fetch runtime status from the configured API target. Start the backend, verify the endpoint, or switch Data source back to Harness in Tweaks while debugging."
        productionDescription="Chat is waiting for the local runtime to become reachable. Keep the page open while the service recovers, or switch Data source back to Harness in Tweaks to inspect sample conversations."
        onRetry={retryLiveData}
        onSwitchToTestData={switchToTestData}
      >
        <ChatLiveLoadingGhost />
      </LiveDataUnavailableOverlay>
    )
  }

  if (showLiveLoading) {
    return <ChatLiveLoadingGhost />
  }

  return (
    <>
      <DestructiveActionDialog
        open={conversationPendingDelete !== null}
        onOpenChange={(open) => {
          if (!open) setConversationPendingDelete(null)
        }}
        title={`Delete "${conversationPendingDelete?.title ?? 'chat'}"?`}
        description="This permanently removes the selected chat and its message history from local storage. This action cannot be undone."
        destructiveLabel="Delete chat"
        onConfirm={confirmDeleteSelectedConversation}
        returnFocusRef={deleteDialogReturnFocusRef}
      />
      <TextInputDialog
        open={systemPromptDialogOpen}
        onOpenChange={updateSystemPromptDialogOpen}
        title="Set system prompt"
        description="Saved instructions are sent before every chat message in this browser. Leave it empty to use the model defaults."
        label="System prompt"
        value={systemPromptDraft}
        onValueChange={setSystemPromptDraft}
        onSave={saveSystemPrompt}
        placeholder="You are a careful mesh-llm operator. Keep answers grounded in the current cluster state."
        saveLabel="Save prompt"
        returnFocusRef={systemPromptButtonRef}
      />
      <AttachmentPreviewDialog
        attachment={selectedAttachmentPreview}
        onOpenChange={(open) => {
          if (!open) setSelectedAttachmentPreview(null)
        }}
      />
      <ChatConversationPanel
        sidebar={sidebar}
        hideSidebar={conversations.conversations.length === 0}
        stickToBottomKey={`${displayedConversationId}:${latestTurnToken}`}
        title={data.title}
        subtitle={activeConversation?.title}
        actions={actions}
        composer={
          <Composer
            key={composerConversationId}
            value={composerDraft.prompt}
            onChange={updateComposerPrompt}
            onAttach={(files) => {
              setFailedSubmission(null)
              handledChatErrorRef.current = null
              pendingRetryRef.current = null
              updateComposerAttachments((current) => [...current, ...files])
            }}
            attachmentCount={composerDraft.attachments.length}
            disabled={composerIsPreparingAttachments || !canChat}
            isPreparingAttachments={composerIsPreparingAttachments}
            preparingStage={attachmentProcessingStatus?.stage}
            preparingAttachmentCount={attachmentProcessingStatus?.attachmentCount ?? 0}
            onSystemPrompt={openSystemPromptDialog}
            onSend={() => void sendPrompt()}
            onStop={stopStreamingResponse}
            onRetry={() => void retryLastResponse()}
            canRetry={canRetry}
            isStreaming={composerIsStreaming}
            sendMode={composerShouldQueue ? 'queue' : 'send'}
            textareaRef={composerTextareaRef}
            systemPromptButtonRef={systemPromptButtonRef}
            showSystemPromptButton={systemPromptButtonEnabled}
            placeholder={canChat ? 'Ask me anything...' : 'Waiting for a warm model...'}
          />
        }
        activeMessages={activeMessages}
        activeModelName={activeModelName}
        conversations={conversations.conversations}
        activeConversationIsStreaming={activeConversationIsStreaming}
        lastActiveMessage={lastActiveMessage}
        displayedConversationId={displayedConversationId}
        submittedAttachmentsByMessageId={submittedAttachmentsByMessageId}
        transparencyTabEnabled={transparencyTabEnabled}
        inspectedMessage={inspectedMessage}
        stoppedConversationIds={stoppedConversationIds}
        visibleAttachmentProcessingStatus={visibleAttachmentProcessingStatus}
        visibleFailedSubmission={visibleFailedSubmission}
        visibleQueuedSubmissions={visibleQueuedSubmissions}
        showStreamingPlaceholder={showStreamingPlaceholder}
        onMessageAreaClick={() => setInspectedMessage(undefined)}
        onInspectMessage={inspectMessage}
        onStopStreaming={stopStreamingResponse}
        onOpenAttachment={setSelectedAttachmentPreview}
        onRemoveQueuedSubmission={removeQueuedSubmission}
      />
    </>
  )
}

export function ChatPage(props: ChatPageProps = {}) {
  const existingSession = useOptionalChatSession()
  if (existingSession) {
    return <ChatPageContent {...props} />
  }

  return (
    <ChatSessionProvider data={props.data ?? CHAT_HARNESS}>
      <ChatPageContent {...props} />
    </ChatSessionProvider>
  )
}
