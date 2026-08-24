import { describe, expect, it } from 'vitest'
import { LogsApiError } from '@/features/logs/api/client'
import type { LoggingStatus } from '@/lib/api/types'
import { resolveLogsSchemaCompatibility } from '@/features/logs/lib/logs-schema-compatibility'

function loggingStatus(overrides: Partial<LoggingStatus> = {}): LoggingStatus {
  return {
    metadata_available: true,
    capture_mode: 'metadata_only',
    artifact_capture_available: true,
    artifact_capture_ready: true,
    ...overrides
  }
}

function schemaError(schemaVersion: number | undefined, supportedSchemaVersion: number | undefined) {
  return new LogsApiError(409, 'logging_schema_incompatible', { schemaVersion, supportedSchemaVersion })
}

describe('resolveLogsSchemaCompatibility', () => {
  it('returns the versions from a schema_incompatible status', () => {
    expect(
      resolveLogsSchemaCompatibility(
        loggingStatus({ metadata_state: 'schema_incompatible', schema_version: 2, supported_schema_version: 1 })
      )
    ).toEqual({ schemaVersion: 2, supportedSchemaVersion: 1 })
  })

  it('falls through to undefined when a schema_incompatible status is missing a version', () => {
    expect(
      resolveLogsSchemaCompatibility(
        loggingStatus({ metadata_state: 'schema_incompatible', supported_schema_version: 1 })
      )
    ).toBeUndefined()
    expect(
      resolveLogsSchemaCompatibility(loggingStatus({ metadata_state: 'schema_incompatible', schema_version: 2 }))
    ).toBeUndefined()
  })

  it('falls through when the status is not schema_incompatible', () => {
    expect(
      resolveLogsSchemaCompatibility(
        loggingStatus({ metadata_state: 'ready', schema_version: 2, supported_schema_version: 1 })
      )
    ).toBeUndefined()
  })

  it('falls through to a matching error when the status path is incomplete', () => {
    expect(
      resolveLogsSchemaCompatibility(
        loggingStatus({ metadata_state: 'schema_incompatible', schema_version: 2 }),
        schemaError(5, 4)
      )
    ).toEqual({ schemaVersion: 5, supportedSchemaVersion: 4 })
  })

  it('falls through to errors when status is undefined', () => {
    expect(resolveLogsSchemaCompatibility(undefined, schemaError(4, 3))).toEqual({
      schemaVersion: 4,
      supportedSchemaVersion: 3
    })
  })

  it('returns the versions from a logging_schema_incompatible LogsApiError', () => {
    expect(resolveLogsSchemaCompatibility(undefined, schemaError(3, 2))).toEqual({
      schemaVersion: 3,
      supportedSchemaVersion: 2
    })
  })

  it('returns undefined for a LogsApiError missing either detail', () => {
    expect(resolveLogsSchemaCompatibility(undefined, schemaError(undefined, 2))).toBeUndefined()
    expect(resolveLogsSchemaCompatibility(undefined, schemaError(3, undefined))).toBeUndefined()
    expect(
      resolveLogsSchemaCompatibility(undefined, new LogsApiError(409, 'logging_schema_incompatible'))
    ).toBeUndefined()
  })

  it('returns undefined for errors that are not LogsApiError', () => {
    expect(resolveLogsSchemaCompatibility(undefined, new Error('boom'))).toBeUndefined()
    expect(resolveLogsSchemaCompatibility(undefined, 'boom')).toBeUndefined()
  })

  it('returns the first matching error when several errors are passed', () => {
    expect(resolveLogsSchemaCompatibility(undefined, schemaError(1, 2), schemaError(3, 4))).toEqual({
      schemaVersion: 1,
      supportedSchemaVersion: 2
    })
  })

  it('returns undefined for no status and no errors', () => {
    expect(resolveLogsSchemaCompatibility(undefined)).toBeUndefined()
  })
})
