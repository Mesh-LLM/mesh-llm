/**
 * Creates an opaque browser-side ID.
 *
 * Secure contexts return a standard UUID from `crypto.randomUUID()` and ignore
 * the prefix. The prefix is applied only in fallback contexts where
 * `crypto.randomUUID()` is unavailable, such as non-secure HTTP origins.
 */
export function createBrowserId(prefix = 'id'): string {
  if (typeof crypto !== 'undefined' && typeof crypto.randomUUID === 'function') {
    return crypto.randomUUID()
  }

  return `${prefix}-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 10)}`
}
