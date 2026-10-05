import { createBrowserId } from '@/lib/api/browser-id'

export function generateRequestId(): string {
  return createBrowserId('request')
}
