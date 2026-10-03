import type {
  ChatHarnessData,
  ConfigurationHarnessData,
  DashboardHarnessData,
  ShellHarnessData,
  TomlValidationWarning
} from '@/features/app-tabs/types'
import { CHAT_THREADS, CONVERSATIONS, TRANSPARENCY_NODES } from './chat-fixtures'
import { CONFIGURATION_DEFAULTS } from './configuration-defaults'
import { CFG_CATALOG, CFG_NODES, INITIAL_ASSIGNS } from './configuration-fixtures'
import { MESH_NODES, MODELS, PEER_SUMMARY, PEERS, STATUS_METRICS } from './dashboard-fixtures'

export const DASHBOARD_HARNESS: DashboardHarnessData = {
  hero: {
    title: 'Your private mesh',
    description:
      'Build personal AI from open models. Pool machines across your home, office, or friends — no cloud needed.',
    actions: [
      { label: 'Learn more', href: 'https://meshllm.cloud/', tone: 'link' },
      { label: 'GitHub', href: 'https://github.com/Mesh-LLM/mesh-llm', tone: 'secondary' }
    ]
  },
  statusMetrics: STATUS_METRICS,
  peers: PEERS,
  peerSummary: PEER_SUMMARY,
  models: MODELS,
  meshNodeSeeds: MESH_NODES,
  meshId: 'dashboard-mesh',
  connect: {
    installHref: 'https://meshllm.cloud/#install',
    apiStatus: 'configured target',
    runCommand: 'mesh-llm --auto --join <mesh-invite-token>',
    description: 'contribute compute to the mesh'
  },
  wakeableNodes: [
    {
      logical_id: 'vast-a100-1',
      state: 'sleeping',
      models: ['Qwen2.5-72B-Instruct'],
      vram_gb: 80,
      provider: 'Vast'
    },
    {
      logical_id: 'runpod-h100-2',
      state: 'waking',
      models: ['DeepSeek-R1', 'Qwen3-32B'],
      vram_gb: 94,
      wake_eta_secs: 420
    }
  ]
}

export const SHELL_HARNESS: ShellHarnessData = {
  productName: 'mesh-llm',
  brand: { primary: 'mesh', accent: 'llm' },
  footerLinks: [{ label: 'Docs', href: 'https://meshllm.cloud/' }],
  footerTrailingLink: { label: 'GitHub', href: 'https://github.com/Mesh-LLM/mesh-llm' },
  topNavApiAccessLinks: [
    { href: 'https://meshllm.cloud/', label: 'Docs' },
    { href: 'https://meshllm.cloud/#install', label: 'Install' }
  ],
  topNavJoinCommands: [
    {
      label: 'Invite token',
      value: '<mesh-invite-token>',
      hint: 'Paste your issued token into any join command below.',
      noWrapValue: true
    },
    {
      label: 'Auto join and serve command',
      value: 'mesh-llm --auto --join <mesh-invite-token>',
      prefix: '$',
      hint: 'Matches the Connect panel flow: join, select a model, and serve the API.'
    },
    { label: 'Client-only join command', value: 'mesh-llm client --join <mesh-invite-token>', prefix: '$' }
  ],
  topNavJoinLinks: [
    { href: 'https://meshllm.cloud/', label: 'Setup' },
    { href: 'https://meshllm.cloud/#install', label: 'Install' },
    { href: 'https://meshllm.cloud/#blackboard', label: 'Blackboard' }
  ]
}

export const CHAT_HARNESS: ChatHarnessData = {
  title: 'Chat',
  conversations: CONVERSATIONS,
  conversationGroups: [
    { title: 'Today', conversationIds: ['c1', 'c2'] },
    { title: 'Earlier', conversationIds: [] }
  ],
  transparencyNodes: TRANSPARENCY_NODES,
  threads: CHAT_THREADS,
  models: MODELS,
  actionMetrics: [
    { id: 'nodes', icon: 'cpu', label: '1 node' },
    { id: 'vram', icon: 'hard-drive', label: '61.7 GB' }
  ],
  modelLabel: 'Model'
}

export const CONFIGURATION_HARNESS: ConfigurationHarnessData = {
  title: 'Configuration',
  description:
    "Drag models from the catalog onto a node's VRAM container. Pooled nodes combine all devices into one bar.",
  nodes: CFG_NODES,
  assigns: INITIAL_ASSIGNS,
  catalog: CFG_CATALOG,
  preferredAssignId: 'a2',
  defaults: CONFIGURATION_DEFAULTS,
  audit: {
    categories: [
      {
        id: 'logs-general',
        label: 'General',
        summary: 'Master enable, summary, and export controls',
        help: 'General request-log settings written to the local config file',
        tomlSection: 'logging',
        order: 10
      },
      {
        id: 'logs-retention',
        label: 'Retention',
        summary: 'How long request logs are kept and when cleanup runs',
        help: 'Retention settings written to the local config file',
        tomlSection: 'logging',
        order: 20
      },
      {
        id: 'logs-buffers',
        label: 'Buffers & Replay',
        summary: 'In-memory event buffers and the replay window',
        help: 'Buffer settings written to the local config file',
        tomlSection: 'logging',
        order: 30
      },
      {
        id: 'logs-artifacts',
        label: 'Artifacts & Storage',
        summary: 'On-disk artifact capture and byte limits',
        help: 'Artifact settings written to the local config file',
        tomlSection: 'logging',
        order: 40
      },
      {
        id: 'logs-webhooks',
        label: 'Webhooks',
        summary: 'Outbound webhook delivery of log events',
        help: 'Webhook settings written to the local config file',
        tomlSection: 'logging',
        order: 50
      },
      {
        id: 'logs-audit',
        label: 'Audit file sink',
        summary: 'Rotating local audit-file output; separate from the durable Logs ledger',
        help: 'These settings control the rotating [logging.audit] file sink. The Logs page reads the durable local ledger.',
        tomlSection: 'logging.audit',
        order: 60
      }
    ],
    settings: [
      {
        id: 'logging.enabled',
        categoryId: 'logs-general',
        canonicalPath: 'logging.enabled',
        tomlSection: 'logging',
        tomlKey: 'enabled',
        settingOrder: 10,
        icon: 'layers',
        label: 'Request logging enabled',
        description: 'Enable or disable request log capture.',
        inheritedLabel: 'Written to the local mesh-llm config file',
        visibility: 'standard' as const,
        mutability: 'restart-required' as const,
        applyMode: 'static_on_load' as const,
        restartScope: 'process_restart' as const,
        valueSchema: { kind: 'boolean' },
        baselineValue: 'on',
        control: {
          kind: 'choice',
          name: 'enabled',
          value: 'on',
          presentation: 'toggle',
          options: [
            { value: 'on', label: 'On' },
            { value: 'off', label: 'Off' }
          ]
        }
      },
      {
        id: 'logging.retention_ttl_secs',
        categoryId: 'logs-retention',
        canonicalPath: 'logging.retention_ttl_secs',
        tomlSection: 'logging',
        tomlKey: 'retention_ttl_secs',
        icon: 'gauge',
        label: 'Log retention period',
        description: 'How long captured request logs are retained.',
        inheritedLabel: 'Written to the local mesh-llm config file',
        visibility: 'standard' as const,
        mutability: 'runtime' as const,
        applyMode: 'dynamic_apply' as const,
        restartScope: 'none' as const,
        valueSchema: { kind: 'integer' },
        baselineValue: '604800',
        control: {
          kind: 'range',
          name: 'retention_ttl_secs',
          value: '604800',
          min: 60,
          max: 604800,
          step: 60,
          unit: 'seconds'
        }
      },
      {
        id: 'logging.replay_capacity',
        categoryId: 'logs-buffers',
        canonicalPath: 'logging.replay_capacity',
        tomlSection: 'logging',
        tomlKey: 'replay_capacity',
        icon: 'server',
        label: 'Replay buffer capacity',
        description: 'Number of recent log entries kept for replay.',
        inheritedLabel: 'Written to the local mesh-llm config file',
        visibility: 'standard' as const,
        mutability: 'runtime' as const,
        applyMode: 'dynamic_apply' as const,
        restartScope: 'none' as const,
        valueSchema: { kind: 'integer' },
        baselineValue: '25',
        control: {
          kind: 'range',
          name: 'replay_capacity',
          value: '25',
          min: 1,
          max: 5000,
          step: 1,
          unit: 'entries'
        }
      },
      {
        id: 'logging.artifact.capture_mode',
        categoryId: 'logs-artifacts',
        canonicalPath: 'logging.artifact.capture_mode',
        tomlSection: 'logging.artifact',
        tomlKey: 'capture_mode',
        icon: 'folder',
        label: 'Artifact capture mode',
        description: 'Control how request artifacts are captured and stored.',
        inheritedLabel: 'Written to the local mesh-llm config file',
        visibility: 'standard' as const,
        mutability: 'restart-required' as const,
        applyMode: 'static_on_load' as const,
        restartScope: 'process_restart' as const,
        valueSchema: { kind: 'enum', values: ['metadata_only', 'redacted_artifacts'] },
        baselineValue: 'metadata_only',
        control: {
          kind: 'choice',
          name: 'capture_mode',
          value: 'metadata_only',
          presentation: 'segmented',
          options: [
            { value: 'metadata_only', label: 'Metadata only' },
            { value: 'redacted_artifacts', label: 'Redacted artifacts' }
          ]
        }
      },
      {
        id: 'logging.webhook.url',
        categoryId: 'logs-webhooks',
        canonicalPath: 'logging.webhook.url',
        tomlSection: 'logging.webhook',
        tomlKey: 'url',
        icon: 'zap',
        label: 'Webhook URL',
        description: 'Destination URL for outbound webhook log delivery.',
        inheritedLabel: 'Written to the local mesh-llm config file',
        visibility: 'standard' as const,
        mutability: 'restart-required' as const,
        applyMode: 'static_on_load' as const,
        restartScope: 'process_restart' as const,
        valueSchema: { kind: 'url' },
        baselineValue: '',
        control: {
          kind: 'text',
          name: 'url',
          value: ''
        }
      },
      {
        id: 'logging.audit.enabled',
        categoryId: 'logs-audit',
        canonicalPath: 'logging.audit.enabled',
        tomlSection: 'logging.audit',
        tomlKey: 'enabled',
        settingOrder: 10,
        icon: 'shield',
        label: 'Audit file sink enabled',
        description:
          'Enable or disable the rotating local audit file sink. This does not erase or replace the durable Logs ledger.',
        inheritedLabel: 'Written to the local mesh-llm config file',
        visibility: 'standard' as const,
        mutability: 'runtime' as const,
        applyMode: 'dynamic_apply' as const,
        restartScope: 'none' as const,
        valueSchema: { kind: 'boolean' },
        baselineValue: 'on',
        control: {
          kind: 'choice',
          name: 'enabled',
          value: 'on',
          presentation: 'toggle',
          options: [
            { value: 'on', label: 'On' },
            { value: 'off', label: 'Off' }
          ]
        }
      }
    ],
    preview: [{ label: 'Generated logs settings', value: '6 settings', meta: 'harness' }]
  },
  configFilePath: '~/.mesh-llm/config.toml',
  validationWarnings: [
    { kind: 'ok', text: 'All pinned models have valid gpu_id targets.' },
    {
      kind: 'warn',
      text: 'carrack · GPU 0 · GLM-4.7-Flash will exceed 80% VRAM at 16K context. Consider 8K or moving to GPU 1.'
    },
    { kind: 'ok', text: 'Plugin endpoint http://localhost:8000/v1 is reachable.' },
    { kind: 'info', text: 'Flash attention is on by default, no per-model override emitted.' }
  ] satisfies TomlValidationWarning[],
  launchSummaryConfig: {
    httpBind: '0.0.0.0:9337',
    mmap: 'off'
  }
}
