import { LogRequestId } from '@/features/logs/api/ids'
import type { LogOutcome, LogRequest, LogSource } from '@/features/logs/api/schemas'
import { compareLogInstants } from '@/features/logs/lib/log-instant'
import { HARNESS_REFERENCE_TIME, harnessTimestamp } from './support'

export { HARNESS_REFERENCE_TIME }

const DIRECT_CALLER_ID = '9f0c4cbe8cb7a8d5d577c20e50ef03fd2f63a2e7fd9897c155823bcbb281bb04'
const RELAY_CALLER_ID = 'de2f01895ab34c2c8f5d97a703311f5c7279082eef191644c397d2175210aa9b'

export const HARNESS_LOG_SCENARIO_IDS = {
  activeStream: LogRequestId.parse('10000000-0000-4000-8000-000000000001'),
  completedMesh: LogRequestId.parse('10000000-0000-4000-8000-000000000002'),
  failedRetry: LogRequestId.parse('10000000-0000-4000-8000-000000000003'),
  rejectedAdmission: LogRequestId.parse('10000000-0000-4000-8000-000000000004'),
  cancelledClient: LogRequestId.parse('10000000-0000-4000-8000-000000000005'),
  droppedCapacity: LogRequestId.parse('10000000-0000-4000-8000-000000000006'),
  completedLocal: LogRequestId.parse('10000000-0000-4000-8000-000000000007'),
  completedSparse: LogRequestId.parse('10000000-0000-4000-8000-000000000008'),
  completedActiveSource: LogRequestId.parse('10000000-0000-4000-8000-000000000009'),
  failedOpaque: LogRequestId.parse('10000000-0000-4000-8000-00000000000a')
} as const

type RequestFixtureInput = {
  readonly requestId: LogRequestId
  readonly outcome: LogOutcome
  readonly createdMinutesAgo: number
  readonly terminalMinutesAgo: number | undefined
  readonly route: string | undefined
  readonly model: string | undefined
  readonly provider: string | undefined
  readonly engine: string | undefined
  readonly statusCode: number | undefined
  readonly source: LogSource
  readonly callerEndpointId?: string
  readonly callerAddr?: string
  readonly callerPathType?: LogRequest['callerPathType']
}

function requestFixture(input: RequestFixtureInput): LogRequest {
  return {
    requestId: input.requestId,
    outcome: input.outcome,
    createdAt: harnessTimestamp(input.createdMinutesAgo),
    terminalAt: input.terminalMinutesAgo === undefined ? undefined : harnessTimestamp(input.terminalMinutesAgo),
    route: input.route,
    model: input.model,
    provider: input.provider,
    engine: input.engine,
    statusCode: input.statusCode,
    source: input.source,
    ...(input.callerEndpointId ? { callerEndpointId: input.callerEndpointId } : {}),
    ...(input.callerAddr ? { callerAddr: input.callerAddr } : {}),
    ...(input.callerPathType ? { callerPathType: input.callerPathType } : {})
  }
}

const CORE_REQUESTS: readonly LogRequest[] = [
  requestFixture({
    requestId: HARNESS_LOG_SCENARIO_IDS.activeStream,
    outcome: 'active',
    createdMinutesAgo: 1,
    terminalMinutesAgo: undefined,
    route: 'chat_completions',
    model: 'Qwen3-8B-Q4_K_M.gguf',
    provider: 'mesh',
    engine: 'raw_ingress',
    statusCode: undefined,
    source: 'active',
    callerEndpointId: DIRECT_CALLER_ID,
    callerAddr: '203.0.113.24:48712',
    callerPathType: 'remote_quic_http'
  }),
  requestFixture({
    requestId: HARNESS_LOG_SCENARIO_IDS.completedMesh,
    outcome: 'completed',
    createdMinutesAgo: 2,
    terminalMinutesAgo: 1.05,
    route: 'chat_completions',
    model: 'Qwen3-30B-A3B-Q4_K_M.gguf',
    provider: 'openai_frontend',
    engine: 'chat_completion_stream',
    statusCode: 200,
    source: 'durable',
    callerEndpointId: DIRECT_CALLER_ID,
    callerAddr: '203.0.113.24:48712',
    callerPathType: 'remote_quic_http'
  }),
  requestFixture({
    requestId: HARNESS_LOG_SCENARIO_IDS.failedRetry,
    outcome: 'failed',
    createdMinutesAgo: 4,
    terminalMinutesAgo: 3,
    route: 'responses',
    model: 'DeepSeek-R1-Distill-Qwen-32B-Q4_K_M.gguf',
    provider: 'openai_frontend',
    engine: 'responses_stream',
    statusCode: 502,
    source: 'durable',
    callerEndpointId: RELAY_CALLER_ID,
    callerPathType: 'relay'
  }),
  requestFixture({
    requestId: HARNESS_LOG_SCENARIO_IDS.rejectedAdmission,
    outcome: 'rejected',
    createdMinutesAgo: 6,
    terminalMinutesAgo: 5.9,
    route: 'management_post',
    model: undefined,
    provider: 'management_api',
    engine: 'management_post',
    statusCode: 400,
    source: 'active'
  }),
  requestFixture({
    requestId: HARNESS_LOG_SCENARIO_IDS.cancelledClient,
    outcome: 'cancelled',
    createdMinutesAgo: 8,
    terminalMinutesAgo: 7,
    route: 'chat_completions',
    model: 'Llama-3.1-8B-Instruct-Q4_K_M.gguf',
    provider: 'openai_frontend',
    engine: 'chat_completion',
    statusCode: 499,
    source: 'active'
  }),
  requestFixture({
    requestId: HARNESS_LOG_SCENARIO_IDS.droppedCapacity,
    outcome: 'dropped',
    createdMinutesAgo: 10,
    terminalMinutesAgo: 9.8,
    route: 'management_get_status',
    model: 'Qwen3-235B-A22B-Q4_K_M.gguf',
    provider: 'management_api',
    engine: 'management_get_status',
    statusCode: 503,
    source: 'durable'
  }),
  requestFixture({
    requestId: HARNESS_LOG_SCENARIO_IDS.completedLocal,
    outcome: 'completed',
    createdMinutesAgo: 12,
    terminalMinutesAgo: 11,
    route: 'completions',
    model: 'Phi-4-mini-instruct-Q4_K_M.gguf',
    provider: 'openai_frontend',
    engine: 'completion',
    statusCode: 200,
    source: 'durable',
    callerAddr: '127.0.0.1:54321',
    callerPathType: 'local_http'
  }),
  requestFixture({
    requestId: HARNESS_LOG_SCENARIO_IDS.completedSparse,
    outcome: 'completed',
    createdMinutesAgo: 18,
    terminalMinutesAgo: 17.5,
    route: undefined,
    model: undefined,
    provider: undefined,
    engine: undefined,
    statusCode: undefined,
    source: 'durable'
  }),
  requestFixture({
    requestId: HARNESS_LOG_SCENARIO_IDS.completedActiveSource,
    outcome: 'completed',
    createdMinutesAgo: 24,
    terminalMinutesAgo: 23,
    route: 'models',
    model: undefined,
    provider: 'openai_frontend',
    engine: 'models',
    statusCode: 200,
    source: 'active'
  }),
  requestFixture({
    requestId: HARNESS_LOG_SCENARIO_IDS.failedOpaque,
    outcome: 'failed',
    createdMinutesAgo: 32,
    terminalMinutesAgo: 31,
    route: 'completions',
    model: undefined,
    provider: 'openai_frontend',
    engine: 'completion_stream',
    statusCode: 500,
    source: 'durable'
  })
]

type VolumeProfile = {
  readonly createdMinutesAgo: number
  readonly outcome: Exclude<LogOutcome, 'active'>
  readonly route: string | undefined
  readonly model: string | undefined
  readonly provider: string | undefined
  readonly engine: string | undefined
  readonly statusCode: number | undefined
  readonly source: LogSource
}

const VOLUME_PROFILES = [
  {
    createdMinutesAgo: 14,
    outcome: 'completed',
    route: 'responses',
    model: 'Gemma-3-12B-Q4_K_M.gguf',
    provider: 'openai_frontend',
    engine: 'responses',
    statusCode: 200,
    source: 'durable'
  },
  {
    createdMinutesAgo: 28,
    outcome: 'failed',
    route: 'chat_completions',
    model: 'Qwen2.5-Coder-14B-Q5_K_M.gguf',
    provider: 'mesh-routed',
    engine: 'skippy',
    statusCode: 503,
    source: 'durable'
  },
  {
    createdMinutesAgo: 45,
    outcome: 'completed',
    route: 'models',
    model: undefined,
    provider: 'openai_frontend',
    engine: 'models',
    statusCode: 200,
    source: 'durable'
  },
  {
    createdMinutesAgo: 63,
    outcome: 'cancelled',
    route: 'responses',
    model: 'Llama-3.2-3B-Instruct-Q4_K_M.gguf',
    provider: 'mesh-routed',
    engine: 'native',
    statusCode: 499,
    source: 'active'
  },
  {
    createdMinutesAgo: 82,
    outcome: 'rejected',
    route: 'chat_completions',
    model: undefined,
    provider: undefined,
    engine: undefined,
    statusCode: 429,
    source: 'active'
  },
  {
    createdMinutesAgo: 105,
    outcome: 'dropped',
    route: 'responses',
    model: 'Qwen3-32B-Q4_K_M.gguf',
    provider: undefined,
    engine: undefined,
    statusCode: 503,
    source: 'durable'
  },
  {
    createdMinutesAgo: 132,
    outcome: 'completed',
    route: 'completions',
    model: 'Phi-3.5-mini-instruct-Q4_K_M.gguf',
    provider: 'local-native',
    engine: 'native',
    statusCode: 200,
    source: 'durable'
  },
  {
    createdMinutesAgo: 165,
    outcome: 'failed',
    route: 'responses',
    model: 'Llama-3.3-70B-Instruct-Q4_K_M.gguf',
    provider: 'mesh-routed',
    engine: 'native',
    statusCode: 504,
    source: 'durable'
  },
  {
    createdMinutesAgo: 205,
    outcome: 'completed',
    route: 'chat_completions',
    model: 'DeepSeek-Coder-V2-Lite-Q4_K_M.gguf',
    provider: 'mesh-routed',
    engine: 'skippy',
    statusCode: 200,
    source: 'durable'
  },
  {
    createdMinutesAgo: 250,
    outcome: 'completed',
    route: 'management_get_models',
    model: undefined,
    provider: 'management_api',
    engine: 'management_get_models',
    statusCode: 200,
    source: 'active'
  },
  {
    createdMinutesAgo: 300,
    outcome: 'failed',
    route: 'completions',
    model: 'Mistral-7B-Instruct-v0.3-Q4_K_M.gguf',
    provider: 'local-native',
    engine: 'native',
    statusCode: 408,
    source: 'durable'
  },
  {
    createdMinutesAgo: 355,
    outcome: 'completed',
    route: 'chat_completions',
    model: 'Qwen3-14B-Q5_K_M.gguf',
    provider: 'mesh-routed',
    engine: 'skippy',
    statusCode: 200,
    source: 'durable'
  },
  {
    createdMinutesAgo: 410,
    outcome: 'cancelled',
    route: 'chat_completions',
    model: 'Gemma-2-9B-it-Q4_K_M.gguf',
    provider: 'mesh-routed',
    engine: 'native',
    statusCode: 499,
    source: 'durable'
  },
  {
    createdMinutesAgo: 470,
    outcome: 'completed',
    route: undefined,
    model: undefined,
    provider: undefined,
    engine: undefined,
    statusCode: 200,
    source: 'durable'
  },
  {
    createdMinutesAgo: 540,
    outcome: 'failed',
    route: 'responses',
    model: 'Command-R7B-Q4_K_M.gguf',
    provider: undefined,
    engine: undefined,
    statusCode: 500,
    source: 'durable'
  },
  {
    createdMinutesAgo: 650,
    outcome: 'completed',
    route: 'chat_completions',
    model: 'Granite-3.1-8B-Instruct-Q4_K_M.gguf',
    provider: 'mesh-routed',
    engine: 'skippy',
    statusCode: 200,
    source: 'active'
  }
] satisfies readonly VolumeProfile[]

const VOLUME_REQUESTS = VOLUME_PROFILES.map((profile, index) =>
  requestFixture({
    requestId: LogRequestId.parse(`20000000-0000-4000-8000-${(index + 1).toString(16).padStart(12, '0')}`),
    outcome: profile.outcome,
    createdMinutesAgo: profile.createdMinutesAgo,
    terminalMinutesAgo: profile.createdMinutesAgo - 1,
    route: profile.route,
    model: profile.model,
    provider: profile.provider,
    engine: profile.engine,
    statusCode: profile.statusCode,
    source: profile.source
  })
)

export const HARNESS_LOG_FIXTURES: readonly LogRequest[] = [...CORE_REQUESTS, ...VOLUME_REQUESTS].sort((left, right) =>
  compareLogInstants(right.createdAt, left.createdAt)
)
