import type { RefObject } from 'react'
import { Cpu, HardDrive } from 'lucide-react'
import { LiveDataUnavailableOverlay } from '@/components/ui/LiveDataUnavailableOverlay'
import { DestructiveActionDialog } from '@/components/ui/DestructiveActionDialog'
import { TextInputDialog } from '@/components/ui/TextInputDialog'
import { ChatLiveLoadingGhost } from '@/features/chat/components/ChatLiveLoadingGhost'
import { ChatSidebar } from '@/features/chat/components/ChatSidebar'
import { Composer } from '@/features/chat/components/Composer'
import { ModelSelect } from '@/features/chat/components/ModelSelect'
import { TransparencyPane } from '@/features/chat/components/transparency/TransparencyPane'
import type {
  ChatActionMetric,
  ChatHarnessData,
  Conversation,
  ConversationGroup,
  ModelSelectOption,
  TransparencyMessage
} from '@/features/app-tabs/types'
import {
  AttachmentPreviewDialog,
  type AttachmentProcessingStatus,
  type SubmittedAttachmentPreview
} from '@/features/chat/pages/chat-page-attachments'
import { ChatConversationPanel, type ChatConversationPanelProps } from '@/features/chat/pages/ChatConversationPanel'
import type {
  ConversationComposerDraft,
  DeleteConversationOptions,
  FailedSubmission,
  QueuedSubmission
} from '@/features/chat/pages/chat-page-submissions'

type SidebarTab = 'conversations' | 'transparency'

type ChatPageLayoutProps = {
  data: ChatHarnessData
  showLiveError: boolean
  showLiveLoading: boolean
  onRetryLiveData: () => void
  onSwitchToTestData: () => void
  sidebarTab: SidebarTab
  onSidebarTabChange: (tab: SidebarTab) => void
  conversations: Conversation[]
  conversationGroups: ConversationGroup[]
  activeConversationId?: string
  messageCounts: Record<string, number>
  streamingConversationIds: readonly string[]
  onSelectConversation: (conversation: Conversation) => void
  onRenameConversation: (conversation: Conversation, title: string) => void
  onDeleteConversation: (conversation: Conversation, options?: DeleteConversationOptions) => void
  onNewChat: () => void
  transparencyTabEnabled: boolean
  inspectedMessage: TransparencyMessage | undefined
  conversationPendingDelete: Conversation | null
  onDeleteDialogOpenChange: (open: boolean) => void
  onConfirmDeleteConversation: () => void
  deleteDialogReturnFocusRef: RefObject<HTMLElement | null>
  systemPromptDialogOpen: boolean
  onSystemPromptDialogOpenChange: (open: boolean) => void
  systemPromptDraft: string
  onSystemPromptDraftChange: (value: string) => void
  onSaveSystemPrompt: (value: string) => void
  systemPromptButtonRef: RefObject<HTMLButtonElement | null>
  selectedAttachmentPreview: SubmittedAttachmentPreview | null
  onAttachmentPreviewOpenChange: (open: boolean) => void
  actionMetrics: ChatActionMetric[]
  modelLabel: string
  modelOptions: ModelSelectOption[]
  selectedModelValue: string
  onModelChange: (value: string) => void
  composerConversationId: string
  composerDraft: ConversationComposerDraft
  onComposerPromptChange: (value: string) => void
  onComposerAttachmentsChange: (files: File[]) => void
  composerAttachmentCount: number
  composerDisabled: boolean
  composerIsPreparingAttachments: boolean
  attachmentProcessingStage: AttachmentProcessingStatus['stage'] | undefined
  attachmentProcessingCount: number
  onOpenSystemPrompt: () => void
  onSendPrompt: () => void
  onStopStreaming: () => void
  onRetryLastResponse: () => void
  canRetry: boolean
  composerIsStreaming: boolean
  composerSendMode: 'send' | 'queue'
  composerTextareaRef: RefObject<HTMLTextAreaElement | null>
  showSystemPromptButton: boolean
  canChat: boolean
  activeConversation: Conversation | undefined
  latestTurnToken: number
  activeMessages: ChatConversationPanelProps['activeMessages']
  activeModelName: string
  activeConversationIsStreaming: boolean
  lastActiveMessage: ChatConversationPanelProps['lastActiveMessage']
  displayedConversationId: string
  submittedAttachmentsByMessageId: Record<string, SubmittedAttachmentPreview[]>
  stoppedConversationIds: Set<string>
  visibleAttachmentProcessingStatus: AttachmentProcessingStatus | null
  visibleFailedSubmission: FailedSubmission | null
  visibleQueuedSubmissions: QueuedSubmission[]
  showStreamingPlaceholder: boolean
  onMessageAreaClick: () => void
  onInspectMessage: (message: TransparencyMessage) => void
  onOpenAttachment: (attachment: SubmittedAttachmentPreview) => void
  onRemoveQueuedSubmission: (submissionId: string) => void
}

function ChatMetricBadge({ metric }: { metric: ChatActionMetric }) {
  const Icon = metric.icon === 'cpu' ? Cpu : HardDrive

  return (
    <span className="hidden shrink-0 items-center gap-1.5 whitespace-nowrap rounded-full border border-border px-2.5 py-0.5 text-[length:var(--density-type-caption)] font-medium text-fg-faint md:inline-flex">
      <Icon className="size-3" /> {metric.label}
    </span>
  )
}

export function ChatPageLayout({
  data,
  showLiveError,
  showLiveLoading,
  onRetryLiveData,
  onSwitchToTestData,
  sidebarTab,
  onSidebarTabChange,
  conversations,
  conversationGroups,
  activeConversationId,
  messageCounts,
  streamingConversationIds,
  onSelectConversation,
  onRenameConversation,
  onDeleteConversation,
  onNewChat,
  transparencyTabEnabled,
  inspectedMessage,
  conversationPendingDelete,
  onDeleteDialogOpenChange,
  onConfirmDeleteConversation,
  deleteDialogReturnFocusRef,
  systemPromptDialogOpen,
  onSystemPromptDialogOpenChange,
  systemPromptDraft,
  onSystemPromptDraftChange,
  onSaveSystemPrompt,
  systemPromptButtonRef,
  selectedAttachmentPreview,
  onAttachmentPreviewOpenChange,
  actionMetrics,
  modelLabel,
  modelOptions,
  selectedModelValue,
  onModelChange,
  composerConversationId,
  composerDraft,
  onComposerPromptChange,
  onComposerAttachmentsChange,
  composerAttachmentCount,
  composerDisabled,
  composerIsPreparingAttachments,
  attachmentProcessingStage,
  attachmentProcessingCount,
  onOpenSystemPrompt,
  onSendPrompt,
  onStopStreaming,
  onRetryLastResponse,
  canRetry,
  composerIsStreaming,
  composerSendMode,
  composerTextareaRef,
  showSystemPromptButton,
  canChat,
  activeConversation,
  latestTurnToken,
  activeMessages,
  activeModelName,
  activeConversationIsStreaming,
  lastActiveMessage,
  displayedConversationId,
  submittedAttachmentsByMessageId,
  stoppedConversationIds,
  visibleAttachmentProcessingStatus,
  visibleFailedSubmission,
  visibleQueuedSubmissions,
  showStreamingPlaceholder,
  onMessageAreaClick,
  onInspectMessage,
  onOpenAttachment,
  onRemoveQueuedSubmission
}: ChatPageLayoutProps) {
  const sidebar = (
    <ChatSidebar
      tab={sidebarTab}
      onTabChange={onSidebarTabChange}
      conversations={conversations}
      conversationGroups={conversationGroups}
      activeId={activeConversationId}
      messageCounts={messageCounts}
      streamingConversationIds={streamingConversationIds}
      onSelectConversation={onSelectConversation}
      onRenameConversation={onRenameConversation}
      onDeleteConversation={onDeleteConversation}
      onNewChat={onNewChat}
      transparency={<TransparencyPane message={inspectedMessage} nodes={data.transparencyNodes} />}
      showTransparency={transparencyTabEnabled}
    />
  )

  const actions = (
    <>
      {actionMetrics.map((metric) => (
        <ChatMetricBadge key={metric.id} metric={metric} />
      ))}
      <div className="flex min-w-0 flex-1 basis-full items-center gap-2 sm:basis-auto md:flex-none">
        <span className="hidden shrink-0 whitespace-nowrap text-[length:var(--density-type-caption)] text-fg-faint md:inline">
          {modelLabel}
        </span>
        <ModelSelect options={modelOptions} value={selectedModelValue} onChange={onModelChange} />
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
        onRetry={onRetryLiveData}
        onSwitchToTestData={onSwitchToTestData}
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
        onOpenChange={onDeleteDialogOpenChange}
        title={`Delete "${conversationPendingDelete?.title ?? 'chat'}"?`}
        description="This permanently removes the selected chat and its message history from local storage. This action cannot be undone."
        destructiveLabel="Delete chat"
        onConfirm={onConfirmDeleteConversation}
        returnFocusRef={deleteDialogReturnFocusRef}
      />
      <TextInputDialog
        open={systemPromptDialogOpen}
        onOpenChange={onSystemPromptDialogOpenChange}
        title="Set system prompt"
        description="Saved instructions are sent before every chat message in this browser. Leave it empty to use the model defaults."
        label="System prompt"
        value={systemPromptDraft}
        onValueChange={onSystemPromptDraftChange}
        onSave={onSaveSystemPrompt}
        placeholder="You are a careful mesh-llm operator. Keep answers grounded in the current cluster state."
        saveLabel="Save prompt"
        returnFocusRef={systemPromptButtonRef}
      />
      <AttachmentPreviewDialog attachment={selectedAttachmentPreview} onOpenChange={onAttachmentPreviewOpenChange} />
      <ChatConversationPanel
        sidebar={sidebar}
        hideSidebar={conversations.length === 0}
        stickToBottomKey={`${displayedConversationId}:${latestTurnToken}`}
        title={data.title}
        subtitle={activeConversation?.title}
        actions={actions}
        composer={
          <Composer
            key={composerConversationId}
            value={composerDraft.prompt}
            onChange={onComposerPromptChange}
            onAttach={onComposerAttachmentsChange}
            attachmentCount={composerAttachmentCount}
            disabled={composerDisabled}
            isPreparingAttachments={composerIsPreparingAttachments}
            preparingStage={attachmentProcessingStage}
            preparingAttachmentCount={attachmentProcessingCount}
            onSystemPrompt={onOpenSystemPrompt}
            onSend={onSendPrompt}
            onStop={onStopStreaming}
            onRetry={onRetryLastResponse}
            canRetry={canRetry}
            isStreaming={composerIsStreaming}
            sendMode={composerSendMode}
            textareaRef={composerTextareaRef}
            systemPromptButtonRef={systemPromptButtonRef}
            showSystemPromptButton={showSystemPromptButton}
            placeholder={canChat ? 'Ask me anything...' : 'Waiting for a warm model...'}
          />
        }
        activeMessages={activeMessages}
        activeModelName={activeModelName}
        conversations={conversations}
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
        onMessageAreaClick={onMessageAreaClick}
        onInspectMessage={onInspectMessage}
        onStopStreaming={onStopStreaming}
        onOpenAttachment={onOpenAttachment}
        onRemoveQueuedSubmission={onRemoveQueuedSubmission}
      />
    </>
  )
}
