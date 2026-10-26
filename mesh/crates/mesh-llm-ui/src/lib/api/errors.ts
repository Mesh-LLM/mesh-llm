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
 * line, which is not a reason and must not be shown as one. The parser marks
 * that fallback explicitly so matching server text remains a valid reason.
 */
export type ParsedApiErrorBody = {
  body: string
  synthesizedFallback: boolean
}

export function apiErrorReason(errorBody: ParsedApiErrorBody): string | undefined {
  const reason = errorBody.body.trim()
  if (!reason || errorBody.synthesizedFallback) return undefined
  return reason
}

export async function parseApiErrorBody(response: Response): Promise<ParsedApiErrorBody> {
  let body = ''
  try {
    body = await response.text()
    const json = JSON.parse(body) as unknown
    if (json && typeof json === 'object') {
      const obj = json as Record<string, unknown>
      if (obj['error'] && typeof obj['error'] === 'object') {
        const err = obj['error'] as Record<string, unknown>
        if (typeof err['message'] === 'string') {
          return { body: err['message'], synthesizedFallback: false }
        }
      }
      if (typeof obj['error'] === 'string') {
        return { body: obj['error'], synthesizedFallback: false }
      }
      if (typeof obj['message'] === 'string') {
        return { body: obj['message'], synthesizedFallback: false }
      }
    }
  } catch {
    // body was not JSON, return as-is
  }
  return {
    body: body || `HTTP ${response.status}: ${response.statusText}`,
    synthesizedFallback: !body
  }
}
