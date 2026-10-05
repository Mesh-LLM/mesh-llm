import { useState } from 'react'
import { Download } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { LogsApiClient } from '@/features/logs/api/client'
import type { AvailableLogArtifact } from '@/features/logs/lib/log-payload-content'

export type LogArtifactDownloadControlProps = {
  readonly artifact: AvailableLogArtifact
}

type DownloadAction = { readonly tone: 'error' | 'success'; readonly message: string } | undefined

function saveArtifactDownload(bytes: Uint8Array, fileName: string, mediaType: string): boolean {
  if (
    typeof document === 'undefined' ||
    typeof URL === 'undefined' ||
    typeof URL.createObjectURL !== 'function' ||
    typeof URL.revokeObjectURL !== 'function'
  ) {
    return false
  }

  const copy = new Uint8Array(bytes.byteLength)
  copy.set(bytes)
  const url = URL.createObjectURL(new Blob([copy.buffer], { type: mediaType }))
  const anchor = document.createElement('a')
  anchor.href = url
  anchor.download = fileName
  anchor.rel = 'noopener'
  anchor.hidden = true
  document.body.append(anchor)
  anchor.click()
  anchor.remove()
  window.setTimeout(() => URL.revokeObjectURL(url), 0)
  return true
}

export function LogArtifactDownloadControl({ artifact }: LogArtifactDownloadControlProps) {
  const [action, setAction] = useState<DownloadAction>()
  const [pending, setPending] = useState(false)

  async function downloadArtifact(): Promise<void> {
    setPending(true)
    setAction(undefined)
    try {
      const result = await new LogsApiClient().downloadArtifact(artifact.artifactId)
      switch (result.state) {
        case 'unavailable':
          setAction({ tone: 'error', message: 'This artifact is no longer available for download.' })
          break
        case 'download':
          setAction(
            saveArtifactDownload(result.download.bytes, result.download.fileName, result.download.mediaType)
              ? { tone: 'success', message: 'Artifact download started.' }
              : { tone: 'error', message: 'This browser cannot save the retained artifact.' }
          )
          break
        default:
          assertNever(result)
      }
    } catch {
      setAction({ tone: 'error', message: 'The retained artifact could not be downloaded.' })
    } finally {
      setPending(false)
    }
  }

  return (
    <div className="flex flex-wrap items-center gap-2">
      <Button
        className="ui-control"
        disabled={pending}
        onClick={() => void downloadArtifact()}
        size="sm"
        type="button"
        variant="outline"
      >
        <Download aria-hidden="true" className="size-3.5" />
        {pending ? 'Preparing download…' : 'Download redacted artifact'}
      </Button>
      {action ? (
        <p className={`type-caption ${action.tone === 'error' ? 'text-bad' : 'text-good'}`} role="status">
          {action.message}
        </p>
      ) : null}
    </div>
  )
}

function assertNever(value: never): never {
  throw new Error(`Unhandled artifact download result: ${String(value)}`)
}
