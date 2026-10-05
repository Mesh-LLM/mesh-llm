import type {
  Conversation,
  Decision,
  ThreadMessage,
  TransparencyMessage,
  TransparencyNode
} from '@/features/app-tabs/types'

export const CONVERSATIONS: Conversation[] = [
  {
    id: 'c1',
    title: 'Routing latency notes',
    subtitle: 'Inspect why TTFT rose in tor-1',
    updatedAt: '09:42',
    createdAt: Date.now(),
    messages: [],
    active: true
  },
  {
    id: 'c2',
    title: 'Model capacity draft',
    subtitle: 'Plan pooled placement for coder stack',
    updatedAt: 'Yesterday',
    createdAt: Date.now(),
    messages: []
  }
]
export const TRANSPARENCY_NODES: TransparencyNode[] = [
  { id: 'desk', label: 'YOU', region: 'local', status: 'online' },
  { id: 'carrack', label: 'CARRACK', region: 'tor-1', status: 'online', isLocal: true },
  { id: 'lemony', label: 'LEMONY-28', region: 'nyc-2', status: 'online' },
  { id: 'lemony-29', label: 'LEMONY-29', region: 'sfo-1', status: 'online' }
]
export const TRANSPARENCY_MESSAGE: TransparencyMessage = {
  kind: 'assistant',
  id: 'msg-a1',
  text: 'Here are three revisions with different tones — playful, serious, and technical. Want me to expand any of them?',
  at: '14:53',
  servedBy: 'lemony',
  route: ['desk', 'carrack', 'lemony'],
  model: 'Qwen3.6-35B-A3B-UD',
  receipt: 'rx-92b7',
  metrics: { rttMs: 1, ttftMs: 312, throughput: '22.4 tok/s', tokens: 148 },
  decisions: [
    { id: 'fit', ok: true, label: 'Qwen3.6-35B-A3B-UD warm', detail: 'lemony-28 · 22.1 GB loaded' },
    { id: 'skip', ok: false, label: 'carrack skipped', detail: 'not enough VRAM headroom · 4.1 GB free' },
    { id: 'link', ok: true, label: 'Link healthy', detail: '0.8ms RTT · 0% loss · 1.2Gbps' },
    {
      id: 'policy',
      ok: true,
      label: 'Prompt > 20 tokens → remote',
      detail: 'policy: route big prompts to dedicated node'
    }
  ],
  trace: [
    { id: 'queue', label: 'Queue', ms: 14, tone: 'neutral' },
    { id: 'route', label: 'Route', ms: 22, tone: 'neutral' },
    { id: 'prefill', label: 'Prefill', ms: 290, tone: 'warn' },
    { id: 'decode', label: 'Decode', ms: 6607, tone: 'good' }
  ]
}

const NEWSLETTER_PROMPT = 'Can you draft three short intro paragraphs for a newsletter about local AI?'
const OUTBOUND_SECURITY: Decision[] = [
  { id: 'encrypted', ok: true, label: 'Encrypted in transit', detail: 'TLS 1.3 · mesh-pki' },
  { id: 'local', ok: true, label: 'Endpoint stays local', detail: '127.0.0.1:9337/v1/chat' },
  { id: 'hops', ok: true, label: 'No third-party hops', detail: 'request never leaves your mesh' },
  { id: 'hash', ok: true, label: 'Content hash', detail: '7c02...913a' }
]

const OUTBOUND_TRANSPARENCY_MESSAGE: TransparencyMessage = {
  kind: 'user',
  id: 'msg-u2',
  text: NEWSLETTER_PROMPT,
  at: '14:53',
  requestId: '7c02...913a',
  dispatch: { picked: 'lemony', candidates: 3, bytes: 184, tokens: 22, model: 'Qwen3.6-35B-A3B-UD' },
  route: ['desk', 'carrack', 'lemony'],
  security: OUTBOUND_SECURITY
}

const HELLO_TRANSPARENCY_MESSAGE: TransparencyMessage = {
  kind: 'assistant',
  id: 'msg-a0',
  text: 'Hello! How can I help you today?',
  at: '14:52',
  servedBy: 'carrack',
  route: ['desk', 'carrack'],
  model: 'Qwen3.6-27B-UD',
  receipt: 'rx-52a1',
  metrics: { rttMs: 1, ttftMs: 170, throughput: '36.8 tok/s', tokens: 10 },
  decisions: [
    { id: 'fit', ok: true, label: 'Qwen3.6-27B-UD warm', detail: 'carrack · 17.6 GB loaded' },
    { id: 'local', ok: true, label: 'Local node selected', detail: 'lowest latency for short prompt' },
    { id: 'link', ok: true, label: 'Link healthy', detail: 'local loopback · 0% loss' },
    { id: 'policy', ok: true, label: 'Short prompt stayed local', detail: 'policy: keep small replies on your node' }
  ],
  trace: [
    { id: 'queue', label: 'Queue', ms: 6, tone: 'neutral' },
    { id: 'route', label: 'Route', ms: 8, tone: 'neutral' },
    { id: 'prefill', label: 'Prefill', ms: 170, tone: 'good' },
    { id: 'decode', label: 'Decode', ms: 260, tone: 'good' }
  ]
}

const CAPACITY_PROMPT = 'Can you sketch a pooled placement plan for the coder stack before tomorrow?'
const CAPACITY_REPLY =
  'Use pooled placement on perseus.local for the small Qwen models, then keep Llama isolated on carrack GPU 1 so context-heavy drafts do not fragment the shared pool.'

const CAPACITY_OUTBOUND_MESSAGE: TransparencyMessage = {
  kind: 'user',
  id: 'msg-c2-u1',
  text: CAPACITY_PROMPT,
  at: 'Yesterday',
  requestId: 'c2a8...41ff',
  dispatch: { picked: 'carrack', candidates: 3, bytes: 152, tokens: 16, model: 'Qwen3.6-27B-UD' },
  route: ['desk', 'carrack'],
  security: OUTBOUND_SECURITY
}

const CAPACITY_REPLY_MESSAGE: TransparencyMessage = {
  kind: 'assistant',
  id: 'msg-c2-a1',
  text: CAPACITY_REPLY,
  at: 'Yesterday',
  servedBy: 'carrack',
  route: ['desk', 'carrack'],
  model: 'Qwen3.6-27B-UD',
  receipt: 'rx-c2a8',
  metrics: { rttMs: 1, ttftMs: 184, throughput: '31.2 tok/s', tokens: 64 },
  decisions: [
    { id: 'fit', ok: true, label: 'Qwen3.6-27B-UD warm', detail: 'carrack · 17.6 GB loaded' },
    {
      id: 'capacity',
      ok: true,
      label: 'Placement data available',
      detail: 'configuration plan references pooled VRAM'
    },
    { id: 'link', ok: true, label: 'Link healthy', detail: 'local loopback · 0% loss' },
    {
      id: 'policy',
      ok: true,
      label: 'Planning reply stayed local',
      detail: 'policy: keep capacity drafts on owner node'
    }
  ],
  trace: [
    { id: 'queue', label: 'Queue', ms: 8, tone: 'neutral' },
    { id: 'route', label: 'Route', ms: 12, tone: 'neutral' },
    { id: 'prefill', label: 'Prefill', ms: 184, tone: 'good' },
    { id: 'decode', label: 'Decode', ms: 1980, tone: 'good' }
  ]
}

export const CHAT_THREADS: Record<string, ThreadMessage[]> = {
  c1: [
    {
      id: 'msg-u1',
      messageRole: 'user',
      timestamp: '14:52',
      model: 'Qwen3.6-27B-UD',
      body: 'hello',
      routeNode: 'carrack',
      inspectMessage: {
        kind: 'user',
        id: 'msg-u1',
        text: 'hello',
        at: '14:52',
        requestId: '5a18...9fd0',
        dispatch: { picked: 'carrack', candidates: 3, bytes: 6, tokens: 1, model: 'Qwen3.6-27B-UD' },
        route: ['desk', 'carrack'],
        security: OUTBOUND_SECURITY
      }
    },
    {
      id: 'msg-a0',
      messageRole: 'assistant',
      timestamp: '14:52',
      model: 'Qwen3.6-27B-UD',
      body: 'Hello! How can I help you today?',
      route: 'carrack',
      routeNode: 'carrack',
      tokens: '10 tok',
      tokPerSec: '36.8 tok/s',
      ttft: '170 ms',
      inspectMessage: HELLO_TRANSPARENCY_MESSAGE,
      inspectLabel: 'Inspect transparency'
    },
    {
      id: 'msg-u2',
      messageRole: 'user',
      timestamp: '14:53',
      model: 'Qwen3.6-35B-A3B-UD',
      body: NEWSLETTER_PROMPT,
      routeNode: 'lemony-28',
      inspectMessage: OUTBOUND_TRANSPARENCY_MESSAGE,
      inspectLabel: 'Inspect outbound route'
    },
    {
      id: 'msg-a1',
      messageRole: 'assistant',
      timestamp: '14:53',
      model: 'Qwen3.6-35B-A3B-UD',
      body: 'Here are three revisions with different tones — playful, serious, and technical. Want me to expand any of them?',
      route: 'lemony-28',
      routeNode: 'lemony-28',
      tokens: '148 tok',
      tokPerSec: '22.4 tok/s',
      ttft: '312 ms',
      inspectMessage: TRANSPARENCY_MESSAGE
    }
  ],
  c2: [
    {
      id: 'msg-c2-u1',
      messageRole: 'user',
      timestamp: 'Yesterday',
      model: 'Qwen3.6-27B-UD',
      body: CAPACITY_PROMPT,
      routeNode: 'carrack',
      inspectMessage: CAPACITY_OUTBOUND_MESSAGE,
      inspectLabel: 'Inspect capacity prompt route'
    },
    {
      id: 'msg-c2-a1',
      messageRole: 'assistant',
      timestamp: 'Yesterday',
      model: 'Qwen3.6-27B-UD',
      body: CAPACITY_REPLY,
      route: 'carrack',
      routeNode: 'carrack',
      tokens: '64 tok',
      tokPerSec: '31.2 tok/s',
      ttft: '184 ms',
      inspectMessage: CAPACITY_REPLY_MESSAGE,
      inspectLabel: 'Inspect capacity reply route'
    }
  ]
}
