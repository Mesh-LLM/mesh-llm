import type { RowData } from '@tanstack/react-table'
import { ChevronLeft, ChevronRight, ChevronsLeft, ChevronsRight } from 'lucide-react'
import { Button } from '@/components/ui/button'
import type { TanStackTable } from '@/components/ui/data-table-features'
import { NativeSelect, type NativeSelectOption } from '@/components/ui/NativeSelect'

const PAGE_SIZES: readonly NativeSelectOption[] = [10, 20, 25, 30, 40, 50].map((size) => ({
  value: String(size),
  label: String(size)
}))

export type DataTablePaginationProps<TData extends RowData> = {
  readonly table: TanStackTable<TData>
}

export function DataTablePagination<TData extends RowData>({ table }: DataTablePaginationProps<TData>) {
  const pageSize = table.state.pagination.pageSize
  const pageCount = table.getPageCount()
  const pageIndex = table.state.pagination.pageIndex
  const canPreviousPage = table.getCanPreviousPage()
  const canNextPage = table.getCanNextPage()

  return (
    <div className="flex flex-wrap items-center justify-end gap-3 px-[var(--panel-x)] py-[var(--panel-y)]">
      <div className="flex flex-wrap items-center gap-3">
        <div className="flex items-center gap-2">
          <span className="type-caption text-fg-faint">Rows per page</span>
          <NativeSelect
            ariaLabel="Rows per page"
            className="min-w-[4.5rem]"
            name="data-table-page-size"
            onValueChange={(value) => {
              const size = Number(value)
              if (Number.isFinite(size) && size > 0) table.setPageSize(size)
            }}
            options={PAGE_SIZES}
            value={String(pageSize)}
          />
        </div>
        <span className="type-caption text-fg-faint">
          Page {pageCount === 0 ? 0 : pageIndex + 1} of {pageCount}
        </span>
        <div className="flex items-center gap-1">
          <Button
            className="ui-control size-8 rounded-[var(--radius)]"
            disabled={!canPreviousPage}
            onClick={() => table.setPageIndex(0)}
            size="icon"
            title="Go to first page"
            variant="outline"
          >
            <span className="sr-only">Go to first page</span>
            <ChevronsLeft className="size-3.5" aria-hidden="true" />
          </Button>
          <Button
            className="ui-control size-8 rounded-[var(--radius)]"
            disabled={!canPreviousPage}
            onClick={() => table.previousPage()}
            size="icon"
            title="Go to previous page"
            variant="outline"
          >
            <span className="sr-only">Go to previous page</span>
            <ChevronLeft className="size-3.5" aria-hidden="true" />
          </Button>
          <Button
            className="ui-control size-8 rounded-[var(--radius)]"
            disabled={!canNextPage}
            onClick={() => table.nextPage()}
            size="icon"
            title="Go to next page"
            variant="outline"
          >
            <span className="sr-only">Go to next page</span>
            <ChevronRight className="size-3.5" aria-hidden="true" />
          </Button>
          <Button
            className="ui-control size-8 rounded-[var(--radius)]"
            disabled={!canNextPage}
            onClick={() => table.setPageIndex(Math.max(table.getPageCount() - 1, 0))}
            size="icon"
            title="Go to last page"
            variant="outline"
          >
            <span className="sr-only">Go to last page</span>
            <ChevronsRight className="size-3.5" aria-hidden="true" />
          </Button>
        </div>
      </div>
    </div>
  )
}
