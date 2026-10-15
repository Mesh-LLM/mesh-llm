// Where the page gets the owner's own facts: this plugin's
// `evidence_records_status` tool. Ported from the mesh-llm console fork's
// `lib/use-your-records.ts`, status query only (the console's stored-text
// probe feeds the `Your prompts` pill, which this page does not show).
import { useQuery } from '@tanstack/react-query'
import { fetchRecordsStatus, type RecordsStatus } from '@/features/capsules/api/recordsClient'

export const RECORDS_STATUS_QUERY_KEY = ['capsules', 'records-status'] as const

/** `null` while unanswered, for sample data, or when the tool can't be reached. */
export function useRecordsStatus({ sample }: { sample: boolean }): RecordsStatus | null {
  const statusQuery = useQuery({
    queryKey: RECORDS_STATUS_QUERY_KEY,
    queryFn: fetchRecordsStatus,
    enabled: !sample,
    refetchInterval: 15_000,
    retry: false
  })
  return statusQuery.data ?? null
}
