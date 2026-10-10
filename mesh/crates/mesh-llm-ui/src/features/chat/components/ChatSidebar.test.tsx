import { isValidElement, type ReactNode, type Ref } from 'react'
import { render } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'
import type { Conversation } from '@/features/app-tabs/types'

const triggerRefs: Ref<unknown>[] = []

vi.mock('@/components/ui/DropdownMenu', async (importOriginal) => {
  const actual = await importOriginal<typeof import('@/components/ui/DropdownMenu')>()
  return {
    ...actual,
    DropdownMenuTrigger: ({ children }: { children: ReactNode }) => {
      if (isValidElement<{ ref?: Ref<unknown> }>(children) && children.props.ref) triggerRefs.push(children.props.ref)
      return <>{children}</>
    }
  }
})

const { ChatSidebar } = await import('@/features/chat/components/ChatSidebar')

const conversation: Conversation = { id: 'c1', title: 'Story', updatedAt: '2026-10-08T00:00:00.000Z' }

function renderSidebar(messageCount: number) {
  return (
    <ChatSidebar
      tab="conversations"
      onTabChange={() => undefined}
      conversations={[conversation]}
      messageCounts={{ c1: messageCount }}
      streamingConversationIds={['c1']}
      transparency={null}
    />
  )
}

describe('ChatSidebar', () => {
  // Regression: a new ref function on every render made Radix re-attach the dropdown anchor on
  // each commit. While a reply streamed, those commits chained until React threw
  // "Maximum update depth exceeded" (#185) and took down the chat view.
  it('keeps the conversation action trigger ref stable while the sidebar re-renders', () => {
    triggerRefs.length = 0
    const { rerender } = render(renderSidebar(1))
    for (let count = 2; count <= 20; count += 1) rerender(renderSidebar(count))

    expect(triggerRefs.length).toBeGreaterThan(1)
    expect(new Set(triggerRefs).size).toBe(1)
  })
})
