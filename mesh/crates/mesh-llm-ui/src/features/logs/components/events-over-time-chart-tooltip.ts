import type { ChartTooltipPayloadItem } from '@/components/ui/chart'
import type { EventVolumeBucket } from '@/features/logs/lib/log-volume'

export function hasVisibleEventVolumeTooltip(payload: readonly ChartTooltipPayloadItem[] | undefined): boolean {
  const bucket = payload?.[0]?.payload as EventVolumeBucket | undefined
  return Boolean(bucket && bucket.total > 0)
}
