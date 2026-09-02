export const DRAFT_MODEL_SPECULATION_DEPENDENCY = {
  settingId: 'speculation-mode',
  condition: (value: string) => value === 'draft'
}

export const THROUGHPUT_TOML_SECTION = 'defaults.throughput'
export const HARDWARE_TOML_SECTION = 'defaults.hardware'
export const MODEL_FIT_TOML_SECTION = 'defaults.model_fit'
export const SKIPPY_TRANSPORT_TOML_SECTION = 'defaults.skippy'
export const REQUEST_DEFAULTS_TOML_SECTION = 'defaults.request_defaults'
export const MULTIMODAL_TOML_SECTION = 'defaults.multimodal'
export const ADVANCED_SERVER_TOML_SECTION = 'defaults.advanced.server'

export const MIROSTAT_MODE_DEPENDENCY = {
  settingId: 'mirostat-mode',
  condition: (value: string) => value !== 'disabled'
}

export const PREFILL_CHUNKING_FIXED_DEPENDENCY = {
  settingId: 'prefill-chunking',
  condition: (value: string) => value === 'fixed'
}

export const PREFILL_CHUNKING_SCHEDULE_DEPENDENCY = {
  settingId: 'prefill-chunking',
  condition: (value: string) => value === 'schedule'
}
