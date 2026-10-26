export class ApiError extends Error {
  constructor(
    public readonly status: number,
    public readonly body: string,
    message: string
  ) {
    super(message)
    this.name = 'ApiError'
  }
}

/**
 * The server's own explanation for a failed request, when its body carried one.
 * `parseApiErrorBody` falls back to a synthesized `HTTP <status>: <statusText>`
 * line, which is not a reason and must not be shown as one.
 */
export function apiErrorReason(body: string, status: number): string | undefined {
  const reason = body.trim()
  if (!reason) return undefined
  if (reason.startsWith(`HTTP ${status}:`)) return undefined
  return reason
}

export async function parseApiErrorBody(response: Response): Promise<string> {
  let body = ''
  try {
    body = await response.text()
    const json = JSON.parse(body) as unknown
    if (json && typeof json === 'object') {
      const obj = json as Record<string, unknown>
      if (obj['error'] && typeof obj['error'] === 'object') {
        const err = obj['error'] as Record<string, unknown>
        if (typeof err['message'] === 'string') return err['message']
      }
      if (typeof obj['error'] === 'string') return obj['error']
      if (typeof obj['message'] === 'string') return obj['message']
    }
  } catch {
    // body was not JSON, return as-is
  }
  return body || `HTTP ${response.status}: ${response.statusText}`
}
