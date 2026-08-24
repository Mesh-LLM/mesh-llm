import { env } from '@/lib/env'

export const APP_STORAGE_KEYS = {
  featureFlagOverrides: `${env.storageNamespace}:feature-flags:v1`,
  chatSystemPrompt: `${env.storageNamespace}:chat-system-prompt:v1`,
  preferences: `${env.storageNamespace}:preferences:v1`
}
