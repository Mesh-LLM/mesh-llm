import type { UIMessage, ModelMessage } from '@tanstack/ai'

import { env } from '@/lib/env'
import { buildResponsesInput } from '@/features/chat/api/build-input'

const GATEWAY_OPENAPI_URL = 'https://www.flushnet.net/api/openai.json'
const ACTIONS_OPENAPI_URL = 'https://www.flushnet.net/api/openapi.json'
const MAX_TOOL_ROUNDS = 3

// Keep the authenticated tool turn compact: Qwen3 runs with a 4K context and
// receives the live canonical tool schema in the same completion. The broader
// onboarding policy is used for ordinary chat, while this prompt governs only
// the existing OpenAI-compatible Gateway tool loop.
const FLUSHNET_TOOL_MODE_PROMPT = [
  'You are Flushnet AI Runner in authenticated tool mode.',
  'Use only the supplied canonical OpenAI tools. The caller injects access_code into every Gateway request; never request, reveal, or repeat it.',
  'Gateway tool results are authoritative. For a module request, follow this exact protocol: gatewayIndex first, gatewaySchema for the requested module second, then gatewayExecute using only the exact action returned by gatewaySchema.',
  'Never infer action names. In particular, do not use nodes.list_nodes; execute the exact nodes action supplied by gatewaySchema.',
  'Never say an access code is invalid, expired, unauthorized, or rejected when the corresponding tool result has HTTP 2xx and JSON ok=true.',
  'After a successful node-list result, report the exact returned nodes, status, runtime_mode, and count.',
  'Do not invent owner data, task history, browser state, modules, or execution results. Do not mention internal implementation names.',
  'For node command execution, use runTerminalCommand. Never send prose to a node. Extract only the executable command string (example: dir). Then use runTerminalCommand. If gatewayExecute returns a job_id, call gatewayStatus or getAsyncJobStatus until a terminal status before answering. Use commandList/listCommands only for history fallback.',
  'After tool results, give a concise final answer based only on those results.'
].join(' ')

type JsonRecord = Record<string, unknown>
type ToolDefinition = { type: 'function'; function: { name: string; description: string; parameters: JsonRecord } }
type OpenAiMessage = JsonRecord

type ToolCall = {
  id: string
  type: 'function'
  function: { name: string; arguments: string }
}

type ToolLoopResult = {
  content: string
  model: string
  toolTrace: Array<{ name: string; ok: boolean; httpStatus: number }>
}

let schemaCache: Promise<{ tools: ToolDefinition[]; operationUrls: Record<string, string>; operationMethods: Record<string, 'GET' | 'POST'> }> | undefined

function textFromMessage(message: UIMessage | ModelMessage): string {
  if ('parts' in message) {
    return message.parts
      .filter((part) => part.type === 'text')
      .map((part) => part.content)
      .join('\n')
      .trim()
  }
  if (typeof message.content === 'string') return message.content.trim()
  if (Array.isArray(message.content)) {
    return message.content
      .filter((part) => part.type === 'text')
      .map((part) => part.content)
      .join('\n')
      .trim()
  }
  return ''
}

export function extractFlushnetAccessCode(messages: Array<UIMessage> | Array<ModelMessage>): string | undefined {
  // The live canonical schema deliberately treats access_code as an opaque string.
  // Keep the label requirement so ordinary chat text cannot accidentally enable
  // privileged tools, but do not impose an invented short-token limit.
  const labelledCode = /(?:flushnet\s+)?(?:access[_\s-]*code|code)\s*(?:is|:|=)?\s*([A-Za-z0-9]+)\b/i
  for (let index = messages.length - 1; index >= 0; index -= 1) {
    const message = messages[index]
    if (!message || message.role !== 'user') continue
    const text = textFromMessage(message)
    const match = text.match(labelledCode)
    if (match?.[1]) return match[1]
  }
  return undefined
}

function resolveRef(value: unknown, schemas: JsonRecord): JsonRecord {
  if (!value || typeof value !== 'object') return { type: 'object', properties: {} }
  const record = value as JsonRecord
  const ref = typeof record.$ref === 'string' ? record.$ref : ''
  const prefix = '#/components/schemas/'
  if (ref.startsWith(prefix)) {
    const resolved = schemas[ref.slice(prefix.length)]
    if (resolved && typeof resolved === 'object') return resolved as JsonRecord
  }
  return record
}

function compactParameters(name: string, schema: JsonRecord): JsonRecord {
  const properties = (schema.properties && typeof schema.properties === 'object' ? schema.properties : {}) as JsonRecord
  // Same Gateway operations as NVIDIA; retain every required field and the practical action fields
  // so the local Qwen stays inside its 4K context window.
  if (name !== 'gatewayExecute') return {
    type: String(schema.type ?? 'object'),
    properties,
    required: Array.isArray(schema.required) ? schema.required : [],
    additionalProperties: schema.additionalProperties ?? false
  }
  const allowed = new Set([
    'access_code', 'module', 'action', 'params', 'node_id', 'command', 'url', 'selector',
    'text', 'query', 'instruction', 'timeout_ms', 'job_id', 'run_id', 'file_id', 'mode'
  ])
  const compact = Object.fromEntries(Object.entries(properties).filter(([key]) => allowed.has(key)))
  return {
    type: 'object',
    properties: compact,
    required: Array.isArray(schema.required) ? schema.required : [],
    additionalProperties: false
  }
}

async function canonicalTools(): Promise<{ tools: ToolDefinition[]; operationUrls: Record<string, string>; operationMethods: Record<string, 'GET' | 'POST'> }> {
  schemaCache ??= (async () => {
    const tools: ToolDefinition[] = []
    
const operationUrls: Record<string, string> = {}
    const operationMethods: Record<string, 'GET' | 'POST'> = {}
    const seen = new Set<string>()

    const addTool = (name: string, description: string, parameters: JsonRecord, url: string, method: 'GET' | 'POST') => {
      if (!name || seen.has(name)) return
      seen.add(name)
      tools.push({ type: 'function', function: { name, description, parameters } })
      operationUrls[name] = url
      operationMethods[name] = method
    }

    const addOpenApiPostTools = async (openapiUrl: string, wanted?: Set<string>) => {
      const response = await fetch(openapiUrl)
      if (!response.ok) throw new Error(`Flushnet tool schema failed: ${response.status}`)
      const openapi = await response.json() as JsonRecord
      const components = (openapi.components && typeof openapi.components === 'object' ? openapi.components : {}) as JsonRecord
      const schemas = (components.schemas && typeof components.schemas === 'object' ? components.schemas : {}) as JsonRecord
      const paths = (openapi.paths && typeof openapi.paths === 'object' ? openapi.paths : {}) as JsonRecord
      for (const [path, rawPathItem] of Object.entries(paths)) {
        if (!rawPathItem || typeof rawPathItem !== 'object') continue
        const pathItem = rawPathItem as JsonRecord
        const post = pathItem.post
        if (post && typeof post === 'object') {
          const spec = post as JsonRecord
          const name = typeof spec.operationId === 'string' ? spec.operationId.trim() : ''
          if (name && (!wanted || wanted.has(name))) {
            const body = (spec.requestBody && typeof spec.requestBody === 'object' ? spec.requestBody : {}) as JsonRecord
            const content = (body.content && typeof body.content === 'object' ? body.content : {}) as JsonRecord
            const json = (content['application/json'] && typeof content['application/json'] === 'object' ? content['application/json'] : {}) as JsonRecord
            const schema = resolveRef(json.schema, schemas)
            addTool(name, String(spec.summary ?? spec.description ?? name), compactParameters(name, schema), `https://www.flushnet.net${path}`, 'POST')
          }
        }
      }
    }

    await addOpenApiPostTools(GATEWAY_OPENAPI_URL)
    await addOpenApiPostTools(ACTIONS_OPENAPI_URL, new Set(['runAsyncJob', 'listCommands', 'commandList', 'nodeStats']))

        addTool('getAsyncJobStatus', 'Get async job status or result. Call this after runAsyncJob or gatewayExecute returns a job_id. Repeat with wait_ms until status is completed, failed, timed_out, or canceled.', {
      type: 'object',
      properties: {
        access_code: { type: 'string' },
        job_id: { type: 'string' },
        wait_ms: { type: 'integer', default: 3000 }
      },
      required: ['access_code', 'job_id'],
      additionalProperties: false
    }, 'https://www.flushnet.net/api/tools/job_status_async.php', 'GET')

    if (!tools.length) throw new Error('Flushnet tool schema has no callable operations')
    return { tools, operationUrls, operationMethods }
  })()
  return schemaCache
}

function toolTurnInput(messages: Array<UIMessage> | Array<ModelMessage>): Array<UIMessage> | Array<ModelMessage> {
  // Access-code authentication is extracted before this point. Keep only the
  // newest user task for each authenticated tool turn so prior catalog/tool
  // output cannot overflow Qwen's 4K context window on a follow-up request.
  for (let index = messages.length - 1; index >= 0; index -= 1) {
    if (messages[index]?.role === 'user') return messages.slice(index, index + 1)
  }
  return messages.slice(-1)
}

function normalizeMessages(input: JsonRecord[]): OpenAiMessage[] {
  return input.map((message) => ({ role: message.role, content: message.content }))
}

function disableQwenThinkingForToolTurn(messages: OpenAiMessage[]): void {
  // Qwen3 otherwise consumes the local 4K completion budget in <think> text
  // before emitting a tool call. This is Qwen's documented chat-template
  // directive and is added only to the private model request, never to the UI.
  for (let index = messages.length - 1; index >= 0; index -= 1) {
    const message = messages[index]
    if (message?.role !== 'user' || typeof message.content !== 'string') continue
    if (!message.content.includes('/no_think')) message.content = `${message.content.trim()}\n/no_think`
    return
  }
}

async function requestJson(url: string, method: 'GET' | 'POST', body: unknown, signal?: AbortSignal): Promise<{ status: number; data: unknown }> {
  let finalUrl = url
  const init: RequestInit = { method, headers: { 'Content-Type': 'application/json', Accept: 'application/json' }, signal }
  if (method === 'GET') {
    const params = new URLSearchParams()
    if (body && typeof body === 'object') {
      for (const [key, value] of Object.entries(body as JsonRecord)) {
        if (value !== undefined && value !== null && value !== '') params.set(key, String(value))
      }
    }
    const qs = params.toString()
    if (qs) finalUrl += (finalUrl.includes('?') ? '&' : '?') + qs
  } else {
    init.body = JSON.stringify(body)
  }
  const response = await fetch(finalUrl, init)
  const raw = await response.text()
  let data: unknown = raw
  try { data = raw ? JSON.parse(raw) : {} } catch { /* Preserve non-JSON gateway errors. */ }
  return { status: response.status, data }
}

async function postJson(url: string, body: unknown, signal?: AbortSignal): Promise<{ status: number; data: unknown }> {
  return requestJson(url, 'POST', body, signal)
}

function assistantMessage(data: unknown): JsonRecord {
  if (!data || typeof data !== 'object') throw new Error('Local model returned invalid JSON')
  const choices = (data as JsonRecord).choices
  if (!Array.isArray(choices) || !choices[0] || typeof choices[0] !== 'object') throw new Error('Local model returned no completion choice')
  const message = (choices[0] as JsonRecord).message
  if (!message || typeof message !== 'object') throw new Error('Local model returned no assistant message')
  return message as JsonRecord
}

function toolCalls(message: JsonRecord): ToolCall[] {
  const raw = message.tool_calls
  if (!Array.isArray(raw)) return []
  return raw.flatMap((item, index) => {
    if (!item || typeof item !== 'object') return []
    const call = item as JsonRecord
    const fn = call.function
    if (!fn || typeof fn !== 'object') return []
    const functionRecord = fn as JsonRecord
    const name = typeof functionRecord.name === 'string' ? functionRecord.name : ''
    if (!name) return []
    const args = typeof functionRecord.arguments === 'string' ? functionRecord.arguments : JSON.stringify(functionRecord.arguments ?? {})
    const id = typeof call.id === 'string' && call.id ? call.id : `call_mesh_webchat_${index}`
    return [{ id, type: 'function' as const, function: { name, arguments: args } }]
  })
}


function terminalJobStatus(status: string): boolean {
  return ['completed','failed','timed_out','canceled','cancelled'].includes(status.toLowerCase())
}

function nestedRecord(value: unknown): JsonRecord {
  return value && typeof value === 'object' && !Array.isArray(value) ? value as JsonRecord : {}
}

function jobFromGatewayResponse(value: unknown): JsonRecord {
  const top = nestedRecord(value)
  const data = nestedRecord(top.data)
  const result = nestedRecord(data.result)
  const directJob = nestedRecord(top.job)
  const dataJob = nestedRecord(data.job)
  const resultJob = nestedRecord(result.job)
  return Object.keys(resultJob).length ? resultJob : Object.keys(dataJob).length ? dataJob : Object.keys(directJob).length ? directJob : result
}

function jobIdFromGatewayResponse(value: unknown): string | undefined {
  const top = nestedRecord(value)
  const data = nestedRecord(top.data)
  const result = nestedRecord(data.result)
  const job = jobFromGatewayResponse(value)
  for (const candidate of [top.job_id, data.job_id, result.job_id, job.job_id]) {
    if (typeof candidate === 'string' && candidate) return candidate
  }
  return undefined
}

function jobStatusFromGatewayResponse(value: unknown): string {
  const top = nestedRecord(value)
  const data = nestedRecord(top.data)
  const result = nestedRecord(data.result)
  const job = jobFromGatewayResponse(value)
  for (const candidate of [top.status, data.status, result.status, job.status]) {
    if (typeof candidate === 'string' && candidate) return candidate
  }
  return ''
}

function resultTextFromGatewayResponse(value: unknown): string {
  const job = jobFromGatewayResponse(value)
  let result = nestedRecord(job.result)
  if (!Object.keys(result).length && typeof job.result_json === 'string') {
    try { result = nestedRecord(JSON.parse(job.result_json)) } catch { /* Keep fallback candidates below. */ }
  }
  const payload = nestedRecord(result.payload)
  const nestedResult = nestedRecord(payload.result)
  const content = Array.isArray(nestedResult.content) ? nestedResult.content : []
  const contentText = content.map((item) => nestedRecord(item).text).filter((item): item is string => typeof item === 'string' && item.trim().length > 0).join('\n').trim()
  for (const candidate of [contentText, payload.stdout, payload.output, result.stdout, result.output, payload.stderr, result.stderr, job.error_text]) {
    if (typeof candidate === 'string' && candidate.trim()) return candidate.trim()
  }
  return ''
}

async function autoPollJob(accessCode: string, jobId: string, signal?: AbortSignal, browserMode=false): Promise<{ status:number; data:unknown }> {
  const waits = browserMode ? new Array(30).fill(5000) : [12000,12000,12000,12000,12000,12000,12000,12000,12000,12000,12000,12000]
  let last: { status:number; data:unknown } = {status:200,data:{job_id:jobId,status:'queued'}}
  for (let index=0; index<waits.length; index+=1) {
    const wait_ms=waits[index] ?? 12000
    last=await requestJson('https://www.flushnet.net/api/tools/gateway_status.php','POST',{access_code:accessCode,job_id:jobId,wait_ms},signal)
    const status = jobStatusFromGatewayResponse(last.data)
    if (status && terminalJobStatus(status)) return last
    // Some Gateway bridge modes return a current state immediately even with wait_ms.
    // Pace retries so a short-running node command can reach a terminal state.
    if (index < waits.length-1) await new Promise<void>((resolve) => setTimeout(resolve, 4000))
  }
  return last
}

function canonicalTerminalCommand(raw: string): string {
  let command = raw.trim()
  command = command.replace(/^```[A-Za-z0-9_-]*\s*|\s*```$/g, '')
  const executionStart = command.match(/\b(?:run|execute|launch)\s+(?:the\s+)?/i)
  if (executionStart?.index !== undefined) command = command.slice(executionStart.index + executionStart[0].length)
  command = command.replace(/\s+(?:command|cmd)\b/ig, '')
  command = command.replace(/\s+(?:and\s+)?(?:show|display|return|print)\s+(?:me\s+)?(?:the\s+)?(?:result|output)(?:\s+(?:here|back))?.*$/i, '')
  command = command.replace(/\s+on\s+(?:node\s+)?(?:node_[A-Za-z0-9._-]+|ext_[A-Za-z0-9._-]+)\b.*$/i, '')
  return command.trim()
}

function recoverToolCallsFromError(data: unknown): ToolCall[] {
  const text = JSON.stringify(data)
  const calls: ToolCall[] = []
  const pattern = /\{"name"\s*:\s*"([^"]+)"\s*,\s*"parameters"\s*:\s*(\{[^;]+?\})\}/g
  let match: RegExpExecArray | null
  let index = 0
  while ((match = pattern.exec(text)) !== null) {
    const name = match[1] ?? ''
    const rawParams = match[2] ?? '{}'
    try {
      const parsed = JSON.parse(rawParams)
      calls.push({
        id: `call_recovered_${index}`,
        type: 'function',
        function: { name, arguments: JSON.stringify(parsed) }
      })
      index += 1
    } catch {
      // Ignore malformed recovered fragment.
    }
  }
  return calls
}

function normalizeToolArgs(name: string, args: JsonRecord): JsonRecord {
  const nested = nestedRecord(args.params)
  const nodeId = typeof args.node_id === 'string' ? args.node_id : typeof nested.node_id === 'string' ? nested.node_id : ''
  const rawCommand = typeof args.command === 'string' ? args.command : typeof nested.command === 'string' ? nested.command : ''
  const command = rawCommand ? canonicalTerminalCommand(rawCommand) : ''
  if (name === 'gatewayExecute' && (args.module === 'nodes' || command || args.action === 'nodes')) {
    if (args.module === 'nodes' && String(args.action ?? '') === 'list') return args
    args.module = 'nodes'
    args.action = command ? 'run_command' : args.action === 'nodes' ? 'run_command' : String(args.action ?? 'list')
    args.params = { ...(nested.node_id ? nested : {}), ...(nodeId ? { node_id: nodeId } : {}), ...(command ? { command } : {}) }
    if (nodeId) args.node_id = nodeId
    if (command) args.command = command
    delete args.mode
    delete args.tool
  }
  if (name === 'runAsyncJob') {
    if (command && nodeId) {
      args.command = command
      args.node_id = nodeId
      delete args.tool
      delete args.action
      delete args.params
      args.mode = 'native'
    }
    if (args.tool === 'sandbox_shell' && nodeId) {
      delete args.tool
      delete args.action
      delete args.params
      args.mode = 'native'
    }
  }
  return args
}

function explicitTerminalRequest(messages: Array<UIMessage> | Array<ModelMessage>): { nodeId: string; command: string } | undefined {
  const latest = textFromMessage(toolTurnInput(messages).slice(-1)[0] as UIMessage | ModelMessage)
  if (/\bbrowser(?:[_\s-]*profile[_\s-]*[123])?\b|\bbrowser\s+module\b/i.test(latest)) return undefined
  if (!/\b(?:run|execute|launch)\b/i.test(latest)) return undefined
  const node = latest.match(/\b(?:on\s+)?node\s+(node_[A-Za-z0-9._-]+|ext_[A-Za-z0-9._-]+)\b/i) || latest.match(/\b(node_[A-Za-z0-9._-]+|ext_[A-Za-z0-9._-]+)\b/i)
  if (!node?.[1]) return undefined
  const command = canonicalTerminalCommand(latest)
  if (!command || /^(?:run|execute|launch)$/i.test(command)) return undefined
  return { nodeId: node[1], command }
}



function explicitBrowserRequest(messages: Array<UIMessage> | Array<ModelMessage>): { module: string; action: string; nodeId: string; url?: string } | undefined {
  const latest = textFromMessage(toolTurnInput(messages).slice(-1)[0] as UIMessage | ModelMessage)
  const lower = latest.toLowerCase()
  const mentionsBrowser = /(browser\s*profile\s*[123]|browser_profile_[123]|browser\s+module|\bbrowser\b)/i.test(lower)
  if (!mentionsBrowser) return undefined
  let module = 'browser'
  const profile = lower.match(/\bbrowser[_\s-]*profile[_\s-]*([123])\b/) || lower.match(/\bprofile[_\s-]*([123])\b/)
  if (profile?.[1]) module = `browser_profile_${profile[1]}`
  const node = latest.match(/\b(node_[A-Za-z0-9._-]+|ext_[A-Za-z0-9._-]+)\b/)
  if (!node?.[1]) return undefined
  let action = 'list_sessions'
  if (/\b(?:open|goto|navigate)\b/.test(lower)) action = 'open_url'
  else if (/\ba11y|accessibility\b/.test(lower)) action = 'a11y'
  else if (/\bscreenshot\b/.test(lower)) action = 'screenshot'
  else if (/\blist\b.*\bsessions\b|\bsessions\b/.test(lower)) action = 'list_sessions'
  const url = latest.match(/https?:\/\/\S+/)?.[0]?.replace(/[),.]+$/, '')
  return { module, action, nodeId: node[1], url }
}

async function runExplicitBrowserRequest(accessCode: string, request: { module: string; action: string; nodeId: string; url?: string }, signal?: AbortSignal): Promise<ToolLoopResult> {
  const trace: ToolLoopResult['toolTrace'] = []
  const body: JsonRecord = {
    access_code: accessCode,
    module: request.module,
    action: request.action,
    node_id: request.nodeId,
    params: { node_id: request.nodeId }
  }
  if (request.url) {
    body.url = request.url
    ;(body.params as JsonRecord).url = request.url
  }
  const queued = await requestJson('https://www.flushnet.net/api/tools/gateway_execute.php', 'POST', body, signal)
  trace.push({ name: 'gatewayExecute', ok: queued.status >= 200 && queued.status < 300, httpStatus: queued.status })
  const qtop = nestedRecord(queued.data)
  if (queued.status < 200 || queued.status >= 300 || qtop.ok === false) {
    return { content: JSON.stringify({browser_request: body, gateway_response: queued.data}, null, 2), model: '', toolTrace: trace }
  }
  const jobId = jobIdFromGatewayResponse(queued.data)
  if (!jobId) return { content: JSON.stringify(queued.data, null, 2), model: '', toolTrace: trace }
  const data = nestedRecord(qtop.data)
  const result = nestedRecord(data.result)
  const pollAfter = Number(result.poll_after_ms ?? 3000)
  if (pollAfter > 0) await new Promise<void>((resolve) => setTimeout(resolve, pollAfter))
  const polled = await autoPollJob(accessCode, jobId, signal, true)
  trace.push({ name: 'gatewayStatus', ok: polled.status >= 200 && polled.status < 300, httpStatus: polled.status })
  const status = jobStatusFromGatewayResponse(polled.data)
  const output = resultTextFromGatewayResponse(polled.data)
  return {
    content: JSON.stringify({ job_id: jobId, final_status: status || 'unknown', queued: queued.data, final: polled.data, output }, null, 2),
    model: '',
    toolTrace: trace
  }
}

async function runExplicitNodeListRequest(accessCode: string, signal?: AbortSignal): Promise<ToolLoopResult> {
  const trace: ToolLoopResult['toolTrace'] = []
  const schema = await requestJson('https://www.flushnet.net/api/tools/gateway_schema.php', 'POST', {
    access_code: accessCode, module: 'nodes'
  }, signal)
  trace.push({ name: 'gatewaySchema', ok: schema.status >= 200 && schema.status < 300, httpStatus: schema.status })
  const listed = await requestJson('https://www.flushnet.net/api/tools/gateway_execute.php', 'POST', {
    access_code: accessCode,
    module: 'nodes',
    action: 'list',
    params: {}
  }, signal)
  trace.push({ name: 'gatewayExecute', ok: listed.status >= 200 && listed.status < 300, httpStatus: listed.status })
  return { content: JSON.stringify(listed.data, null, 2), model: '', toolTrace: trace }
}

async function runExplicitTerminalRequest(accessCode: string, request: { nodeId: string; command: string }, signal?: AbortSignal): Promise<ToolLoopResult> {
  const trace: ToolLoopResult['toolTrace'] = []
  const schema = await requestJson('https://www.flushnet.net/api/tools/gateway_schema.php', 'POST', {
    access_code: accessCode, module: 'nodes'
  }, signal)
  trace.push({ name: 'gatewaySchema', ok: schema.status >= 200 && schema.status < 300, httpStatus: schema.status })
  const schemaData = nestedRecord(schema.data)
  if (schema.status < 200 || schema.status >= 300 || schemaData.ok === false) {
    return { content: `Node tool schema request failed: ${JSON.stringify(schema.data)}`, model: '', toolTrace: trace }
  }
  const queued = await requestJson('https://www.flushnet.net/api/tools/gateway_execute.php', 'POST', {
    access_code: accessCode,
    module: 'nodes',
    action: 'run_command',
    node_id: request.nodeId,
    command: request.command,
    timeout_ms: 20000,
    params: { node_id: request.nodeId, command: request.command, timeout_ms: 20000 }
  }, signal)
  trace.push({ name: 'gatewayExecute', ok: queued.status >= 200 && queued.status < 300, httpStatus: queued.status })
  const jobId = jobIdFromGatewayResponse(queued.data)
  if (!jobId) return { content: `Command request was not accepted: ${JSON.stringify(queued.data)}`, model: '', toolTrace: trace }
  const polled = await autoPollJob(accessCode, jobId, signal, false)
  trace.push({ name: 'gatewayStatus', ok: polled.status >= 200 && polled.status < 300, httpStatus: polled.status })
  const status = jobStatusFromGatewayResponse(polled.data)
  const output = resultTextFromGatewayResponse(polled.data)
  if (status === 'completed') return { content: `✅ Command executed successfully on ${request.nodeId}\n\nCommand\n\n${request.command}\n\nResult\n\n${output || '(no stdout)'}`, model: '', toolTrace: trace }
  if (status && terminalJobStatus(status)) return { content: `❌ Command ${status} on ${request.nodeId}\n\n${output || 'No error text was returned.'}`, model: '', toolTrace: trace }
  return { content: `Job ${jobId} is still ${status || 'queued'}.`, model: '', toolTrace: trace }
}

export async function runFlushnetToolChat(
  messages: Array<UIMessage> | Array<ModelMessage>,
  model: string,
  clientId: string,
  requestId: string,
  accessCode: string,
  _systemPrompt: string,
  signal?: AbortSignal
): Promise<ToolLoopResult> {
  // browser requests always bypass model and terminal adapters
  const browserDirect = explicitBrowserRequest(messages)
  if (browserDirect) {
    const browserResult = await runExplicitBrowserRequest(accessCode, browserDirect, signal)
    return { ...browserResult, model }
  }
  const latestText = textFromMessage(toolTurnInput(messages).slice(-1)[0] as UIMessage | ModelMessage).toLowerCase()
  if (latestText.includes('node') && !latestText.includes('run ') && !latestText.includes('execute ')) {
    const listResult = await runExplicitNodeListRequest(accessCode, signal)
    return { ...listResult, model }
  }
  const direct = explicitTerminalRequest(messages)
  if (direct) {
    const directResult = await runExplicitTerminalRequest(accessCode, direct, signal)
    return { ...directResult, model }
  }
  const { tools, operationUrls, operationMethods } = await canonicalTools()
  // Do not repeat the long onboarding prompt inside tool mode: it crowds the
  // live OpenAPI tools out of the local 4K context. Gateway authorization is
  // unchanged and access_code is still injected below for every request.
  const source = await buildResponsesInput(toolTurnInput(messages), model, clientId, requestId, '')
  const openAiMessages: OpenAiMessage[] = normalizeMessages(source.input as unknown as JsonRecord[])
if (model.toLowerCase().includes('qwen')) {
    disableQwenThinkingForToolTurn(openAiMessages)
  }
  openAiMessages.unshift({ role: 'system', content: FLUSHNET_TOOL_MODE_PROMPT })
  const trace: ToolLoopResult['toolTrace'] = []
  let finalContent = ''
  let lastToolResult = ''
  const verifiedTerminalResults: string[] = []

  for (let round = 0; round <= MAX_TOOL_ROUNDS; round += 1) {
    const modelResponse = await postJson(`${env.managementApiUrl}/v1/chat/completions`, {
      model,
      messages: openAiMessages,
      tools,
      tool_choice: 'auto',
      parallel_tool_calls: false,
      temperature: 0.2,
      top_p: 0.7,
      max_tokens: 2048,
      stream: false
    }, signal)
    let assistant: JsonRecord = {}
    let calls: ToolCall[] = []
    if (modelResponse.status < 200 || modelResponse.status >= 300) {
      calls = recoverToolCallsFromError(modelResponse.data)
      if (!calls.length) {
        throw new Error(`Local tool request failed (${model}): ${modelResponse.status} body=${JSON.stringify(modelResponse.data)}`)
      }
      assistant = { content: '', tool_calls: calls }
    } else {
      assistant = assistantMessage(modelResponse.data)
      calls = toolCalls(assistant)
    }
    openAiMessages.push({
      role: 'assistant',
      content: typeof assistant.content === 'string' ? assistant.content : '',
      tool_calls: calls
    })
    if (!calls.length) {
      finalContent = typeof assistant.content === 'string' ? assistant.content : ''
      break
    }
    for (const call of calls) {
      let args: JsonRecord = {}
      try {
        const parsed = JSON.parse(call.function.arguments)
        if (parsed && typeof parsed === 'object' && !Array.isArray(parsed)) args = parsed as JsonRecord
      } catch { /* Gateway receives an explicit invalid-JSON result below. */ }
      args = normalizeToolArgs(call.function.name, args)
      args.access_code = accessCode
      if (call.function.name === 'runTerminalCommand') { args.module='nodes'; args.action='run_command'; call.function.name='gatewayExecute' as any }
      const url = operationUrls[call.function.name]
      const method = operationMethods[call.function.name] ?? 'POST'
      const result = url
        ? await requestJson(url, method, args, signal)
        : { status: 400, data: { ok: false, error: 'unknown_gateway_operation' } }
      trace.push({ name: call.function.name, ok: result.status >= 200 && result.status < 300, httpStatus: result.status })
      openAiMessages.push({
        role: 'tool',
        tool_call_id: call.id,
        name: call.function.name,
        content: JSON.stringify(result.data)
      })
      lastToolResult = JSON.stringify(result.data, null, 2)

      const jobId = jobIdFromGatewayResponse(result.data)
      if (jobId && call.function.name === 'gatewayExecute') {
        const rd = nestedRecord(result.data)
        const dd = nestedRecord(rd.data)
        const rr = nestedRecord(dd.result)
        const pollAfter = Number(rr.poll_after_ms ?? 0)
        if (pollAfter > 0) await new Promise<void>((resolve)=>setTimeout(resolve, pollAfter))
        const browserMode = String(args.module ?? '').startsWith('browser')
        const polled = await autoPollJob(accessCode, jobId, signal, browserMode)
        trace.push({ name: 'getAsyncJobStatus', ok: polled.status >= 200 && polled.status < 300, httpStatus: polled.status })
        openAiMessages.push({
          role: 'tool',
          tool_call_id: call.id,
          name: 'getAsyncJobStatus',
          content: JSON.stringify(polled.data)
        })
        const status = jobStatusFromGatewayResponse(polled.data)
        const output = resultTextFromGatewayResponse(polled.data)
        if (status === 'completed' && output) verifiedTerminalResults.push(`Job ${jobId} completed:
${output}`)
        if (status && terminalJobStatus(status) && status !== 'completed' && output) verifiedTerminalResults.push(`Job ${jobId} ${status}:
${output}`)
      }
    }
  }
  return { content: verifiedTerminalResults.join('\n\n') || finalContent || lastToolResult || 'Tool run completed.', model, toolTrace: trace }
}
