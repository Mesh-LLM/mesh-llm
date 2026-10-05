import { useState } from 'react'
import { Pager } from '@/components/ui/Pager'
import type { LogLifecycleEvent } from '@/features/logs/api/schemas'
import { lifecycleNodes } from './log-request-lifecycle-data'
import { nodesPerPage, sparseListStyle } from './log-request-lifecycle-layout'
import { LifecycleNodeItem } from './LogRequestLifecycleNode'
import { useTrackWidth } from './useLifecycleTrackWidth'

export function LogRequestLifecycleStrip({ events }: { readonly events: readonly LogLifecycleEvent[] }) {
  const nodes = lifecycleNodes(events)
  const { trackRef, width } = useTrackWidth()
  const [page, setPage] = useState(0)

  const perPage = nodesPerPage(width)
  const pageCount = Math.max(1, Math.ceil(nodes.length / perPage))
  const activePage = Math.min(page, pageCount - 1)
  const startIndex = activePage * perPage
  const pageNodes = nodes.slice(startIndex, startIndex + perPage)
  const listStyle = sparseListStyle(pageNodes.length)
  const hasPreviousPage = activePage > 0
  const hasNextPage = activePage < pageCount - 1

  return (
    <div className="-mx-[var(--panel-x)] min-w-0 w-[calc(100%+2*var(--panel-x))] py-5">
      <div className="min-w-0 px-[var(--panel-x)]" data-testid="lifecycle-viewport" ref={trackRef}>
        <ol aria-label="Lifecycle events" className="mx-auto flex min-w-0" style={listStyle}>
          {pageNodes.map((node, index) => (
            <LifecycleNodeItem
              continuationElapsed={index === 0 && hasPreviousPage ? node.elapsed : undefined}
              key={node.key}
              nextElapsed={nodes[startIndex + index + 1]?.elapsed}
              node={node}
              showConnector={index < pageNodes.length - 1}
              showIncoming={index === 0 && hasPreviousPage}
              showOutgoing={index === pageNodes.length - 1 && hasNextPage}
            />
          ))}
        </ol>
      </div>
      <Pager
        ariaLabel="Lifecycle timeline pages"
        className="mt-4"
        count={pageCount}
        nextLabel="Later lifecycle events"
        pageLabel={(index) => `Lifecycle events page ${index + 1} of ${pageCount}`}
        previousLabel="Earlier lifecycle events"
        value={activePage}
        onValueChange={setPage}
      />
    </div>
  )
}
