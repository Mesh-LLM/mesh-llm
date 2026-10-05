import { Database } from 'lucide-react'
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert'
import { StatusBadge } from '@/components/ui/StatusBadge'
import type { LogsSchemaCompatibility } from '@/features/logs/lib/logs-schema-compatibility'

export function LogsSchemaCompatibilityAlert({ schemaVersion, supportedSchemaVersion }: LogsSchemaCompatibility) {
  return (
    <Alert className="panel-shell rounded-[var(--radius)] border-warn/40 bg-warn/5 p-[var(--panel-x)]" role="alert">
      <div className="flex items-start gap-3">
        <Database aria-hidden="true" className="mt-0.5 size-4 shrink-0 text-warn" />
        <div className="min-w-0">
          <AlertTitle className="type-panel-title text-foreground">Log database version mismatch</AlertTitle>
          <AlertDescription className="type-caption mt-1 text-fg-dim">
            This MeshLLM build cannot safely open the local log database schema. Update MeshLLM or restore the build
            that created the database, then restart the node. The database was left unchanged, and inference remains
            available.
          </AlertDescription>
          <div className="mt-3 flex flex-wrap gap-2" aria-label="Log schema compatibility">
            <StatusBadge size="caption" tone="warn">
              Database schema v{schemaVersion}
            </StatusBadge>
            <StatusBadge size="caption" tone="muted">
              Runtime supports v{supportedSchemaVersion}
            </StatusBadge>
          </div>
        </div>
      </div>
    </Alert>
  )
}
