import { createBrowserId } from '@/lib/api/browser-id'

const CLIENT_ID_KEY = 'mesh-llm:client-id'

export function getClientId(): string {
  let id = localStorage.getItem(CLIENT_ID_KEY)
  if (!id) {
    id = createBrowserId('client')
    localStorage.setItem(CLIENT_ID_KEY, id)
  }
  return id
}
