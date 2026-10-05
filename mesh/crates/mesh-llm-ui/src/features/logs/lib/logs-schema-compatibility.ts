import { LogsApiError } from '@/features/logs/api/client'
import type { LoggingStatus } from '@/lib/api/types'

export type LogsSchemaCompatibility = {
  readonly schemaVersion: number
  readonly supportedSchemaVersion: number
}

export function resolveLogsSchemaCompatibility(
  status: LoggingStatus | undefined,
  ...errors: readonly unknown[]
): LogsSchemaCompatibility | undefined {
  if (
    status?.metadata_state === 'schema_incompatible' &&
    status.schema_version !== undefined &&
    status.supported_schema_version !== undefined
  ) {
    return {
      schemaVersion: status.schema_version,
      supportedSchemaVersion: status.supported_schema_version
    }
  }

  for (const error of errors) {
    if (
      error instanceof LogsApiError &&
      error.code === 'logging_schema_incompatible' &&
      error.details?.schemaVersion !== undefined &&
      error.details.supportedSchemaVersion !== undefined
    ) {
      return {
        schemaVersion: error.details.schemaVersion,
        supportedSchemaVersion: error.details.supportedSchemaVersion
      }
    }
  }
  return undefined
}
