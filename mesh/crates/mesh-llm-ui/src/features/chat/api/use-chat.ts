import { useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react'
import { useChat } from '@tanstack/ai-react'
import type { UseChatReturn } from '@tanstack/ai-react'
import type { ThreadMessage } from '@/features/app-tabs/types'
import { threadMessagesToUIMessages } from '@/features/chat/api/use-chat-messages'
import { createMeshConnectionAdapter } from '@/features/chat/api/mesh-connection'
import type { ChatResponseMetadata } from '@/features/chat/api/response-metadata'

type UseMeshChatOptions = {
  conversationId: string
  model: string
  systemPrompt?: string
  /** A node endpoint id: when set, requests go to that node only. */
  target?: string
  /** Restrict this chat's requests to free hosts (`mesh_payment: free_only`). */
  freeOnly?: boolean
  initialMessages: ThreadMessage[]
  onResponseMetadata?: (metadata: ChatResponseMetadata) => void
}

function createMutableStringSource(initialValue: string) {
  let current = initialValue

  return {
    get value() {
      return current
    },
    setValue(value: string) {
      current = value
    }
  }
}

export function useMeshChat({
  conversationId,
  model,
  systemPrompt = '',
  target = '',
  freeOnly = false,
  initialMessages,
  onResponseMetadata
}: UseMeshChatOptions): UseChatReturn {
  const previousConversationIdRef = useRef(conversationId)
  const [currentModel] = useState(() => createMutableStringSource(model))
  const [currentSystemPrompt] = useState(() => createMutableStringSource(systemPrompt))
  const [currentTarget] = useState(() => createMutableStringSource(target))
  const [currentFreeOnly] = useState(() => createMutableStringSource(freeOnly ? 'free_only' : ''))

  useLayoutEffect(() => {
    currentModel.setValue(model)
    currentSystemPrompt.setValue(systemPrompt)
    currentTarget.setValue(target)
    currentFreeOnly.setValue(freeOnly ? 'free_only' : '')
  }, [currentModel, currentSystemPrompt, currentTarget, currentFreeOnly, model, systemPrompt, target, freeOnly])

  const connection = useMemo(
    () =>
      createMeshConnectionAdapter(
        currentModel,
        onResponseMetadata,
        currentSystemPrompt,
        currentTarget,
        currentFreeOnly
      ),
    [currentModel, currentSystemPrompt, currentTarget, currentFreeOnly, onResponseMetadata]
  )
  const hydratedMessages = useMemo(() => threadMessagesToUIMessages(initialMessages), [initialMessages])
  const chat = useChat({ threadId: conversationId, connection, initialMessages: hydratedMessages })

  useEffect(() => {
    const conversationChanged = previousConversationIdRef.current !== conversationId
    previousConversationIdRef.current = conversationId

    if (conversationChanged || (chat.messages.length === 0 && hydratedMessages.length > 0)) {
      chat.setMessages(hydratedMessages)
    }
  }, [chat, conversationId, hydratedMessages])

  return chat
}
