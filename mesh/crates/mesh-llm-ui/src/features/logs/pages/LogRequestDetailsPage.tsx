import { Navigate, useParams, useSearch } from '@tanstack/react-router'
import { LogRequestId } from '@/features/logs/api/ids'
import { closeLogInspector, legacyRequestInspectorSearch } from '@/features/logs/lib/log-search'

export function LogRequestDetailsPage() {
  const { requestId: requestIdParam } = useParams({ from: '/logs/$requestId' })
  const search = useSearch({ from: '/logs/$requestId' })
  const requestId = LogRequestId.tryParse(requestIdParam)
  const nextSearch = requestId ? legacyRequestInspectorSearch(requestId.toString(), search) : closeLogInspector(search)
  return <Navigate replace search={nextSearch} to="/logs" />
}
