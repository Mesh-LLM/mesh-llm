import type { ReactNode } from 'react'
import { MessageSquareMore } from 'lucide-react'
import { EmptyState } from '@/components/ui/EmptyState'
import { MessageRow } from '@/features/chat/components/MessageRow'
import { ChatLayout } from '@/features/chat/layouts/ChatLayout'
import type { Conversation, ThreadMessage, TransparencyMessage } from '@/features/app-tabs/types'
import {
  AttachmentProcessingPanel,
  type AttachmentProcessingStatus,
  type SubmittedAttachmentPreview
} from '@/features/chat/pages/chat-page-attachments'
import {
  getQueuedSubmissionBody,
  getSubmissionBody,
  type FailedSubmission,
  type QueuedSubmission
} from '@/features/chat/pages/chat-page-submissions'

export type ChatConversationPanelProps = {
  sidebar: ReactNode
  hideSidebar: boolean
  stickToBottomKey: string
  title: string
  subtitle?: string
  actions: ReactNode
  composer: ReactNode
  activeMessages: ThreadMessage[]
  activeModelName: string
  conversations: Conversation[]
  activeConversationIsStreaming: boolean
  lastActiveMessage: ThreadMessage | undefined
  displayedConversationId: string
  submittedAttachmentsByMessageId: Record<string, SubmittedAttachmentPreview[]>
  transparencyTabEnabled: boolean
  inspectedMessage: TransparencyMessage | undefined
  stoppedConversationIds: Set<string>
  visibleAttachmentProcessingStatus: AttachmentProcessingStatus | null
  visibleFailedSubmission: FailedSubmission | null
  visibleQueuedSubmissions: QueuedSubmission[]
  showStreamingPlaceholder: boolean
  onMessageAreaClick: () => void
  onInspectMessage: (message: TransparencyMessage) => void
  onStopStreaming: () => void
  onOpenAttachment: (attachment: SubmittedAttachmentPreview) => void
  onRemoveQueuedSubmission: (submissionId: string) => void
}

export function ChatConversationPanel({
  sidebar,
  hideSidebar,
  stickToBottomKey,
  title,
  subtitle,
  actions,
  composer,
  activeMessages,
  activeModelName,
  conversations,
  activeConversationIsStreaming,
  lastActiveMessage,
  displayedConversationId,
  submittedAttachmentsByMessageId,
  transparencyTabEnabled,
  inspectedMessage,
  stoppedConversationIds,
  visibleAttachmentProcessingStatus,
  visibleFailedSubmission,
  visibleQueuedSubmissions,
  showStreamingPlaceholder,
  onMessageAreaClick,
  onInspectMessage,
  onStopStreaming,
  onOpenAttachment,
  onRemoveQueuedSubmission
}: ChatConversationPanelProps) {
  return (
    <ChatLayout
      sidebar={sidebar}
      hideSidebar={hideSidebar}
      stickToBottomKey={stickToBottomKey}
      title={title}
      subtitle={subtitle}
      actions={actions}
      composer={composer}
      onMessageAreaClick={onMessageAreaClick}
    >
      {activeMessages.length === 0 &&
      !showStreamingPlaceholder &&
      visibleQueuedSubmissions.length === 0 &&
      !visibleAttachmentProcessingStatus &&
      !visibleFailedSubmission ? (
        <EmptyState
          tone="accent"
          icon={<MessageSquareMore aria-hidden={true} className="size-10" strokeWidth={1.4} />}
          title="Start Chatting"
          description={
            conversations.length === 0 ? (
              'Type a message below to begin. Your chats stay in this browser, and the mesh routes requests automatically.'
            ) : (
              <>
                No messages yet. Send a message to begin a fresh conversation; replies use{' '}
                <span className="font-mono text-fg">{activeModelName}</span> unless you choose another model.
              </>
            )
          }
        />
      ) : null}
      {activeMessages.map((message) => {
        const transparencyMessage = message.inspectMessage
        const messageAttachments = submittedAttachmentsByMessageId[message.id] ?? []
        const isLatestAssistantMessage = message.messageRole === 'assistant' && message.id === lastActiveMessage?.id
        const messageIsStreamingResponse = activeConversationIsStreaming && isLatestAssistantMessage
        const messageWasStopped = stoppedConversationIds.has(displayedConversationId) && isLatestAssistantMessage
        return (
          <MessageRow
            key={message.id}
            messageRole={message.messageRole}
            timestamp={message.timestamp}
            model={message.model}
            state={messageIsStreamingResponse ? 'streaming' : messageWasStopped ? 'stopped' : 'default'}
            body={message.body}
            route={message.route}
            routeNode={message.routeNode}
            showRouteMetadata={transparencyTabEnabled}
            tokens={message.tokens}
            tokPerSec={message.tokPerSec}
            ttft={message.ttft}
            inspect={
              transparencyTabEnabled && transparencyMessage ? () => onInspectMessage(transparencyMessage) : undefined
            }
            inspectLabel={message.inspectLabel}
            inspected={transparencyMessage != null && inspectedMessage?.id === transparencyMessage.id}
            onStopStreaming={onStopStreaming}
            attachments={messageAttachments.map((attachment) => ({
              id: attachment.id,
              label: attachment.label,
              kind: attachment.kind,
              fileName: attachment.fileName,
              onOpen: () => onOpenAttachment(attachment)
            }))}
          />
        )
      })}
      {visibleAttachmentProcessingStatus ? (
        <AttachmentProcessingPanel status={visibleAttachmentProcessingStatus} />
      ) : null}
      {visibleFailedSubmission ? (
        <>
          {visibleFailedSubmission.includeUserRow ? (
            <MessageRow
              key={`${visibleFailedSubmission.id}-user`}
              messageRole="user"
              timestamp={visibleFailedSubmission.timestamp}
              model={visibleFailedSubmission.model}
              state="default"
              body={getSubmissionBody(visibleFailedSubmission)}
              showRouteMetadata={false}
            />
          ) : null}
          <MessageRow
            key={`${visibleFailedSubmission.id}-error`}
            messageRole="assistant"
            timestamp={visibleFailedSubmission.timestamp}
            model={visibleFailedSubmission.model}
            state="error"
            body={visibleFailedSubmission.errorMessage}
            showRouteMetadata={false}
          />
        </>
      ) : null}
      {showStreamingPlaceholder ? (
        <MessageRow
          key="streaming-response-placeholder"
          messageRole="assistant"
          timestamp="Now"
          model={activeModelName}
          state="streaming"
          body=""
          showRouteMetadata={false}
          onStopStreaming={onStopStreaming}
        />
      ) : null}
      {visibleQueuedSubmissions.map((submission) => (
        <MessageRow
          key={submission.id}
          messageRole="user"
          timestamp={submission.timestamp}
          model="Queued"
          state="queued"
          body={getQueuedSubmissionBody(submission)}
          showRouteMetadata={false}
          onRemoveQueued={() => onRemoveQueuedSubmission(submission.id)}
        />
      ))}
    </ChatLayout>
  )
}
