export class SafeError extends Error {}
export function ensure(condition, message) {
  if (!condition) throw new SafeError(message)
}
export function safeFailure(error) {
  return error instanceof SafeError ? error.message : 'Wallet operation failed; inspect local setup or RPC availability.'
}
