import { fireEvent, render, screen } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'
import { ChatPage } from '@/features/chat/components/ChatPage'

vi.mock('@/components/ui/select', async () => {
  const React = await import('react')

  function MockSelectItem(_props: { value: string; children: React.ReactNode }) {
    return null
  }

  function collectItems(children: React.ReactNode): Array<{ value: string; label: string }> {
    const items: Array<{ value: string; label: string }> = []

    React.Children.forEach(children, (child) => {
      if (!React.isValidElement(child)) return

      if (child.type === MockSelectItem) {
        const props = child.props as { value: string; children: React.ReactNode }
        items.push({
          value: props.value,
          label: String(props.children)
        })
        return
      }

      const props = child.props as { children?: React.ReactNode }
      if (props && 'children' in props && props.children) {
        items.push(...collectItems(props.children))
      }
    })

    return items
  }

  const SelectContext = React.createContext<{
    value?: string
    onValueChange?: (value: string) => void
    items: Array<{ value: string; label: string }>
  } | null>(null)

  function Select({
    value,
    onValueChange,
    children
  }: {
    value?: string
    onValueChange?: (value: string) => void
    children: React.ReactNode
  }) {
    const items = collectItems(children)

    return <SelectContext.Provider value={{ value, onValueChange, items }}>{children}</SelectContext.Provider>
  }

  function SelectTrigger({ className, ...props }: React.SelectHTMLAttributes<HTMLSelectElement>) {
    const context = React.useContext(SelectContext)

    return (
      <select
        {...props}
        className={className}
        value={context?.value ?? ''}
        onChange={(event) => context?.onValueChange?.(event.target.value)}
      >
        {context?.items.map((item) => (
          <option key={item.value} value={item.value}>
            {item.label}
          </option>
        ))}
      </select>
    )
  }

  return {
    Select,
    SelectContent: ({ children }: { children: React.ReactNode }) => <>{children}</>,
    SelectGroup: ({ children }: { children: React.ReactNode }) => <>{children}</>,
    SelectItem: MockSelectItem,
    SelectLabel: () => null,
    SelectSeparator: () => null,
    SelectTrigger,
    SelectValue: () => null
  }
})

function buildProps(overrides: Partial<Parameters<typeof ChatPage>[0]> = {}): Parameters<typeof ChatPage>[0] {
  return {
    status: {
      node_id: 'node-1',
      token: 'invite-token',
      node_state: 'serving',
      node_status: 'Serving',
      is_host: true,
      is_client: false,
      llama_ready: true,
      api_port: 9337,
      model_name: 'model-a',
      model_size_gb: 1,
      inflight_requests: 0,
      my_vram_gb: 12,
      peers: []
    },
    invitationReady: true,
    isPublicMesh: false,
    isFlyHosted: false,
    inflightRequests: 0,
    warmModels: ['model-a'],
    meshModelByName: {},
    modelStatsByName: {},
    selectedModel: 'model-a',
    setSelectedModel: vi.fn(),
    selectedModelNodeCount: 1,
    selectedModelVramGb: 12,
    selectedModelAudio: true,
    selectedModelMultimodal: true,
    composerError: null,
    setComposerError: vi.fn(),
    attachmentSendIssue: null,
    attachmentPreparationMessage: null,
    pendingAttachments: [],
    setPendingAttachments: vi.fn(),
    conversations: [
      {
        id: 'chat-1',
        title: 'Chat 1',
        createdAt: Date.now(),
        updatedAt: String(Date.now()),
        messages: []
      }
    ],
    activeConversationId: 'chat-1',
    onConversationCreate: vi.fn(),
    onConversationSelect: vi.fn(),
    onConversationRename: vi.fn(),
    onConversationDelete: vi.fn(),
    onConversationsClear: vi.fn(),
    messages: [],
    reasoningOpen: {},
    setReasoningOpen: vi.fn(),
    chatScrollRef: { current: null },
    input: '',
    setInput: vi.fn(),
    isSending: false,
    queuedText: null,
    canChat: true,
    canRegenerate: false,
    onStop: vi.fn(),
    onRegenerate: vi.fn(),
    onSubmit: vi.fn(),
    ...overrides
  }
}

describe('ChatPage', () => {
  it('keeps private mesh invitation details token-free', () => {
    render(<ChatPage {...buildProps({ invitationReady: true, selectedModel: 'model-a' })} />)

    expect(screen.getByText('Private mesh invitation ready')).toBeInTheDocument()
    expect(screen.getByText('Selected model: model-a')).toBeInTheDocument()
    expect(screen.getByText('Use the mesh connection controls to securely add another machine.')).toBeInTheDocument()
    expect(screen.queryByText('invite-token')).not.toBeInTheDocument()
    expect(screen.queryByRole('button', { name: /copy invite/i })).not.toBeInTheDocument()
  })

  it('allows attachment-only sends and renders attachment controls', () => {
    render(
      <ChatPage
        {...buildProps({
          pendingAttachments: [
            {
              id: 'att-1',
              kind: 'file',
              dataUrl: 'data:text/plain;base64,aGVsbG8=',
              mimeType: 'text/plain',
              fileName: 'hello.txt',
              status: 'pending'
            }
          ]
        })}
      />
    )

    expect(screen.getByTestId('chat-file-input')).toBeInTheDocument()
    expect(screen.getByTestId('chat-image-input')).toBeInTheDocument()
    expect(screen.getByTestId('chat-audio-input')).toBeInTheDocument()
    expect(screen.getByTestId('chat-send')).toBeEnabled()
    expect(screen.getByText('hello.txt')).toBeInTheDocument()
  })

  it('renders attachment policy errors', () => {
    render(
      <ChatPage
        {...buildProps({
          attachmentSendIssue:
            'Selected model does not support the attached media. Choose a compatible model or remove the attachment.'
        })}
      />
    )

    expect(screen.getByTestId('composer-error')).toHaveTextContent(
      'Selected model does not support the attached media.'
    )
  })

  it('shows attachment preparation progress and disables send', () => {
    render(
      <ChatPage
        {...buildProps({
          attachmentPreparationMessage: 'Preparing PDF in browser…',
          pendingAttachments: [
            {
              id: 'att-pdf',
              kind: 'file',
              dataUrl: 'data:application/pdf;base64,abc',
              mimeType: 'application/pdf',
              fileName: 'scan.pdf',
              status: 'uploading'
            }
          ]
        })}
      />
    )

    expect(screen.getByText('Preparing PDF in browser…')).toBeInTheDocument()
    expect(screen.getByTestId('chat-send')).toBeDisabled()
  })

  it('shows failed image-description state with retry affordance', () => {
    render(
      <ChatPage
        {...buildProps({
          pendingAttachments: [
            {
              id: 'att-image-failed',
              kind: 'image',
              dataUrl: 'data:image/png;base64,abc',
              mimeType: 'image/png',
              fileName: 'legacy.png',
              status: 'failed',
              extractionSummary: 'Image description failed — retry or send placeholder text',
              error: 'Image description failed: model init failed'
            }
          ]
        })}
      />
    )

    expect(screen.getByText('Retry')).toBeInTheDocument()
    expect(screen.getByText('Image description failed: model init failed')).toBeInTheDocument()
    expect(screen.getByText('Image description failed — retry or send placeholder text')).toBeInTheDocument()
  })

  it('shows Queue button label and calls onSubmit when isSending=true', () => {
    const onSubmit = vi.fn()
    render(<ChatPage {...buildProps({ isSending: true, input: 'next message', onSubmit })} />)

    const btn = screen.getByTestId('chat-send')
    expect(btn).toHaveTextContent('Queue')
    fireEvent.click(btn)
    expect(onSubmit).toHaveBeenCalled()
  })

  it('renders queued bubble with the queued text when queuedText is set', () => {
    render(
      <ChatPage
        {...buildProps({
          isSending: true,
          queuedText: 'queued message',
          messages: [
            {
              id: 'msg-1',
              role: 'user' as const,
              content: 'hello'
            }
          ]
        })}
      />
    )

    expect(screen.getByText('Queued')).toBeInTheDocument()
    expect(screen.getByText('queued message')).toBeInTheDocument()
  })

  it('shows Send button and no queued bubble when not sending', () => {
    render(<ChatPage {...buildProps({ isSending: false, queuedText: null })} />)

    expect(screen.getByTestId('chat-send')).toHaveTextContent('Send')
    expect(screen.queryByText('Queued')).not.toBeInTheDocument()
  })

  it('calls onSubmit for attachment-only queue (empty text, pending attachment, isSending=true)', () => {
    const onSubmit = vi.fn()
    render(
      <ChatPage
        {...buildProps({
          isSending: true,
          input: '',
          queuedText: '',
          pendingAttachments: [
            {
              id: 'att-2',
              kind: 'image',
              dataUrl: 'data:image/png;base64,abc',
              mimeType: 'image/png',
              fileName: 'photo.png',
              status: 'pending'
            }
          ],
          onSubmit
        })}
      />
    )

    const btn = screen.getByTestId('chat-send')
    expect(btn).toHaveTextContent('Queue')
    fireEvent.click(btn)
    expect(onSubmit).toHaveBeenCalled()
  })
})
