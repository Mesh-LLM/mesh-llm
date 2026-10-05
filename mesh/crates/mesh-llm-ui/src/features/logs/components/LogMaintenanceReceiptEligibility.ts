import type { LogCleanupReceipt, LogDeleteReceipt } from '@/features/logs/api/schemas'

export function hasRetryableArtifactWork(receipt: LogCleanupReceipt | LogDeleteReceipt) {
  return receipt.state === 'partial' && receipt.artifactDeletion.failed > 0
}

export function canRetryDeleteReceipt(receipt: LogDeleteReceipt) {
  return receipt.state === 'pending' || hasRetryableArtifactWork(receipt)
}
