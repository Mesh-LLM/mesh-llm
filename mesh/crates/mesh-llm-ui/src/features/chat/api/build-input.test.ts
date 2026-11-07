import type { ModelMessage, UIMessage } from '@tanstack/ai'
import { beforeEach, describe, expect, it, vi } from 'vitest'

const uploadAttachmentMock = vi.hoisted(() => vi.fn())

vi.mock('@/features/chat/api/upload-attachment', () => ({
  uploadAttachment: uploadAttachmentMock
}))

import { buildResponsesInput } from '@/features/chat/api/build-input'

function createMessage(parts: UIMessage['parts']): UIMessage {
  return {
    id: 'user-1',
    role: 'user',
    createdAt: new Date('2026-05-06T00:00:00.000Z'),
    parts
  }
}

describe('buildResponsesInput', () => {
  beforeEach(() => {
    uploadAttachmentMock.mockReset()
  })

  it('uploads audio and generic file parts with the shared request id and client id', async () => {
    uploadAttachmentMock.mockResolvedValueOnce('audio-token').mockResolvedValueOnce('file-token')

    const request = await buildResponsesInput(
      [
        createMessage([
          { type: 'text', content: 'Inspect these attachments' },
          {
            type: 'audio',
            source: { type: 'data', value: 'YXVkaW8=', mimeType: 'audio/mpeg' },
            metadata: { fileName: 'note.mp3' }
          },
          {
            type: 'document',
            source: { type: 'data', value: 'ZmlsZQ==', mimeType: 'application/octet-stream' },
            metadata: { fileName: 'notes.bin' }
          }
        ])
      ],
      'model-a',
      'client-123',
      'request-456'
    )

    expect(uploadAttachmentMock).toHaveBeenNthCalledWith(1, {
      requestId: 'request-456',
      bytesBase64: 'YXVkaW8=',
      mimeType: 'audio/mpeg',
      fileName: 'note.mp3'
    })
    expect(uploadAttachmentMock).toHaveBeenNthCalledWith(2, {
      requestId: 'request-456',
      bytesBase64: 'ZmlsZQ==',
      mimeType: 'application/octet-stream',
      fileName: 'notes.bin'
    })
    expect(request.input).toEqual([
      {
        role: 'user',
        content: [
          { type: 'input_text', text: 'Inspect these attachments' },
          { type: 'input_audio', audio_url: 'mesh://blob/client-123/audio-token' },
          {
            type: 'input_file',
            url: 'mesh://blob/client-123/file-token',
            mime_type: 'application/octet-stream',
            file_name: 'notes.bin'
          }
        ]
      }
    ])
  })

  it('preserves simple text messages as a string payload', async () => {
    const request = await buildResponsesInput(
      [createMessage([{ type: 'text', content: 'Hello mesh' }])],
      'model-a',
      'client-123',
      'request-456'
    )

    expect(uploadAttachmentMock).not.toHaveBeenCalled()
    expect(request.input).toEqual([{ role: 'user', content: 'Hello mesh' }])
  })

  it.each(['png', 'jpeg', 'webp'])('summarizes generated %s pixels only in assistant history', async (format) => {
    const image = `![Generated cat](data:image/${format};base64,${'YWJj'.repeat(32_768)})`
    const assistant = {
      ...createMessage([{ type: 'text', content: `Here is your cat.\n\n${image}` }]),
      role: 'assistant' as const
    }
    const user = createMessage([{ type: 'text', content: `Keep this user text: ${image}` }])
    const request = await buildResponsesInput([assistant, user], 'model-a', 'client-123', 'request-456')

    expect(request.input).toEqual([
      { role: 'assistant', content: 'Here is your cat.\n\n[Generated cat]' },
      { role: 'user', content: `Keep this user text: ${image}` }
    ])
    expect(assistant.parts[0]).toEqual({ type: 'text', content: `Here is your cat.\n\n${image}` })
  })

  it('summarizes both string and multipart assistant model messages without changing other media', async () => {
    const image = '![Generated image](data:image/png;base64,YWJj)'
    const messages: ModelMessage[] = [
      { role: 'assistant', content: `First: ${image}` },
      {
        role: 'assistant',
        content: [
          { type: 'text', content: `Second: ${image}` },
          { type: 'image', source: { type: 'url', value: 'https://example.com/reference.png' } }
        ]
      }
    ]
    const request = await buildResponsesInput(messages, 'model-a', 'client-123', 'request-456')

    expect(request.input).toEqual([
      { role: 'assistant', content: 'First: [Generated image]' },
      {
        role: 'assistant',
        content: [
          { type: 'input_text', text: 'Second: [Generated image]' },
          { type: 'input_image', image_url: 'https://example.com/reference.png' }
        ]
      }
    ])
    expect(JSON.stringify(request)).not.toContain('base64')
  })

  it('does not echo image bytes from reference, titled, angle-delimited or incomplete assistant Markdown', async () => {
    const url = 'data:image/png;base64,YWJj'
    const messages: ModelMessage[] = [
      {
        role: 'assistant',
        content: [
          `![Reference][image]\n\n[image]: ${url}`,
          `![Title](${url} "Image title")`,
          `![Angle](<${url}>)`,
          `![Unfinished](${url}`
        ].join('\n\n')
      }
    ]
    const request = await buildResponsesInput(messages, 'model-a', 'client-123', 'request-456')

    expect(JSON.stringify(request)).not.toContain('base64')
    expect(request.input[0]?.content).toContain('Reference')
    expect(request.input[0]?.content).toContain('[Generated image]')
  })

  it('prepends the saved system prompt as a responses system message', async () => {
    const request = await buildResponsesInput(
      [createMessage([{ type: 'text', content: 'Explain the cluster status' }])],
      'model-a',
      'client-123',
      'request-456',
      'Answer like a mesh-llm operator.'
    )

    expect(request.input).toEqual([
      { role: 'system', content: 'Answer like a mesh-llm operator.' },
      { role: 'user', content: 'Explain the cluster status' }
    ])
  })

  it('passes through url-backed audio and document parts without reuploading', async () => {
    const request = await buildResponsesInput(
      [
        createMessage([
          { type: 'audio', source: { type: 'url', value: 'mesh://blob/client-123/audio-token' } },
          {
            type: 'document',
            source: { type: 'url', value: 'mesh://blob/client-123/file-token', mimeType: 'text/plain' },
            metadata: { fileName: 'notes.txt' }
          }
        ])
      ],
      'model-a',
      'client-123',
      'request-456'
    )

    expect(uploadAttachmentMock).not.toHaveBeenCalled()
    expect(request.input).toEqual([
      {
        role: 'user',
        content: [
          { type: 'input_audio', audio_url: 'mesh://blob/client-123/audio-token' },
          {
            type: 'input_file',
            url: 'mesh://blob/client-123/file-token',
            mime_type: 'text/plain',
            file_name: 'notes.txt'
          }
        ]
      }
    ])
  })

  it('converts image parts to backend-compatible image_url blocks without uploading', async () => {
    const request = await buildResponsesInput(
      [
        createMessage([
          { type: 'image', source: { type: 'data', value: 'aW1hZ2U=', mimeType: 'image/png' } },
          { type: 'image', source: { type: 'url', value: 'mesh://blob/client-123/image-token' } }
        ])
      ],
      'model-a',
      'client-123',
      'request-456'
    )

    expect(uploadAttachmentMock).not.toHaveBeenCalled()
    expect(request.input).toEqual([
      {
        role: 'user',
        content: [
          { type: 'input_image', image_url: 'data:image/png;base64,aW1hZ2U=' },
          { type: 'input_image', image_url: 'mesh://blob/client-123/image-token' }
        ]
      }
    ])
  })
})
