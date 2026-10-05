export const statusKeys = {
  all: ['status'] as const,
  detail: () => [...statusKeys.all, 'detail'] as const
}

export const modelKeys = {
  all: ['models'] as const,
  catalog: () => [...modelKeys.all, 'catalog'] as const
}

export const pluginKeys = {
  all: ['plugins'] as const,
  list: () => [...pluginKeys.all, 'list'] as const,
  webUi: (pluginName: string) => [...pluginKeys.all, 'web-ui', pluginName] as const,
  webUiConfig: (pluginName: string) => [...pluginKeys.all, 'web-ui-config', pluginName] as const
}

export const chatKeys = {
  all: ['chat'] as const,
  conversations: () => [...chatKeys.all, 'conversations'] as const,
  messages: (id: string) => [...chatKeys.all, 'messages', id] as const
}
