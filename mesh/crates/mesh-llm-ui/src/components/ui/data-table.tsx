import { startTransition, useEffect, useMemo, useState, type ComponentPropsWithoutRef, type ReactNode } from 'react'
import {
  type ColumnFiltersState,
  type ColumnVisibilityState,
  type PaginationState,
  type RowData,
  type SortingState,
  flexRender,
  useTable
} from '@tanstack/react-table'
import { Search } from 'lucide-react'
import { cn } from '@/lib/cn'
import { Input } from '@/components/ui/input'
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table'
import { DataTablePagination } from '@/components/ui/data-table-pagination'
import {
  dataTableFeatures,
  type ColumnDef,
  type DataTableFeatures,
  type TanStackTable
} from '@/components/ui/data-table-features'

export type DataTableProps<TData extends RowData> = {
  readonly columns: ColumnDef<TData>[]
  readonly data: TData[]
  readonly ariaLabel?: string
  readonly children?: (table: TanStackTable<TData>) => ReactNode
  readonly className?: string
  readonly defaultPageSize?: number
  readonly emptyMessage?: string
  readonly enablePagination?: boolean
  readonly filterColumnId?: string
  readonly filterPlaceholder?: string
  readonly footerClassName?: string | undefined
  readonly getRowId?: (row: TData) => string
  readonly getRowAriaLabel?: (row: TData) => string
  readonly onRowActivate?: (row: TData) => void
  readonly tableClassName?: string
  readonly tableWrapperClassName?: string
}

export function DataTable<TData extends RowData>({
  columns,
  data,
  ariaLabel,
  children,
  className,
  defaultPageSize = 10,
  emptyMessage = 'No results.',
  enablePagination = false,
  filterColumnId,
  filterPlaceholder = 'Filter...',
  footerClassName,
  getRowId,
  getRowAriaLabel,
  onRowActivate,
  tableClassName,
  tableWrapperClassName
}: DataTableProps<TData>) {
  const [sorting, setSorting] = useState<SortingState>([])
  const [columnFilters, setColumnFilters] = useState<ColumnFiltersState>([])
  const [columnVisibility, setColumnVisibility] = useState<ColumnVisibilityState>({})
  const [pagination, setPagination] = useState<PaginationState>({ pageIndex: 0, pageSize: defaultPageSize })

  const tableOptions = useMemo(
    () => ({
      features: dataTableFeatures,
      data,
      columns,
      getRowId,
      state: {
        sorting,
        columnFilters,
        columnVisibility,
        pagination
      },
      onSortingChange: setSorting,
      onColumnFiltersChange: setColumnFilters,
      onColumnVisibilityChange: setColumnVisibility,
      onPaginationChange: setPagination,
      autoResetPageIndex: false,
      manualPagination: !enablePagination
    }),
    [columnFilters, columnVisibility, columns, data, enablePagination, getRowId, pagination, sorting]
  )
  const rowModelTable = useTable<DataTableFeatures, TData>(tableOptions)
  const filteredRowCount = rowModelTable.getFilteredRowModel().rows.length
  const lastPageIndex = Math.max(Math.ceil(filteredRowCount / pagination.pageSize) - 1, 0)
  const effectiveTableOptions = useMemo(() => {
    if (pagination.pageIndex <= lastPageIndex) return tableOptions
    const effectivePagination = { ...pagination, pageIndex: lastPageIndex }
    return {
      ...tableOptions,
      state: { ...tableOptions.state, pagination: effectivePagination }
    }
  }, [lastPageIndex, pagination, tableOptions])
  const table = useTable<DataTableFeatures, TData>(effectiveTableOptions)

  useEffect(() => {
    const nextPageIndex = Math.max(Math.ceil(filteredRowCount / pagination.pageSize) - 1, 0)
    if (pagination.pageIndex <= nextPageIndex) return

    startTransition(() => setPagination((current) => ({ ...current, pageIndex: nextPageIndex })))
  }, [filteredRowCount, pagination.pageIndex, pagination.pageSize])

  const filterValue = filterColumnId ? ((table.getColumn(filterColumnId)?.getFilterValue() as string) ?? '') : undefined

  return (
    <div className={cn('relative w-full', className)}>
      {children?.(table)}
      {filterColumnId ? (
        <div className="flex items-center gap-2 border-b border-border-soft px-[var(--panel-x)] py-2">
          <Search className="size-3.5 shrink-0 text-fg-faint" aria-hidden="true" />
          <Input
            aria-label={filterPlaceholder}
            className="ui-control h-8 max-w-xs rounded-[var(--radius)] text-[length:var(--density-type-caption)]"
            onChange={(event) => table.getColumn(filterColumnId)?.setFilterValue(event.target.value)}
            placeholder={filterPlaceholder}
            value={filterValue}
          />
        </div>
      ) : null}
      <Table aria-label={ariaLabel} className={tableClassName} wrapperClassName={tableWrapperClassName}>
        <TableHeader className="bg-panel-strong">
          {table.getHeaderGroups().map((headerGroup) => (
            <TableRow className="border-border-soft hover:bg-panel-strong" key={headerGroup.id}>
              {headerGroup.headers.map((header) => (
                <TableHead className="type-label h-9 px-3 text-fg-faint" key={header.id}>
                  {header.isPlaceholder ? null : flexRender(header.column.columnDef.header, header.getContext())}
                </TableHead>
              ))}
            </TableRow>
          ))}
        </TableHeader>
        <TableBody>
          {table.getRowModel().rows.length ? (
            table.getRowModel().rows.map((row) => (
              <TableRow
                aria-label={getRowAriaLabel?.(row.original)}
                className={cn(
                  'border-border-soft hover:bg-panel-strong',
                  onRowActivate &&
                    'cursor-pointer outline-none focus-visible:bg-panel-strong focus-visible:ring-2 focus-visible:ring-inset focus-visible:ring-accent'
                )}
                data-state={row.getIsSelected() && 'selected'}
                key={row.id}
                onClick={onRowActivate ? () => onRowActivate(row.original) : undefined}
                onKeyDown={
                  onRowActivate
                    ? (event) => {
                        if (event.target !== event.currentTarget || (event.key !== 'Enter' && event.key !== ' ')) return
                        event.preventDefault()
                        onRowActivate(row.original)
                      }
                    : undefined
                }
                tabIndex={onRowActivate ? 0 : undefined}
              >
                {row.getVisibleCells().map((cell) => (
                  <TableCell className="px-3 py-2" key={cell.id}>
                    {flexRender(cell.column.columnDef.cell, cell.getContext())}
                  </TableCell>
                ))}
              </TableRow>
            ))
          ) : (
            <TableRow className="border-border-soft">
              <TableCell
                className="h-24 text-center text-fg-dim"
                colSpan={Math.max(table.getVisibleLeafColumns().length, 1)}
              >
                {emptyMessage}
              </TableCell>
            </TableRow>
          )}
        </TableBody>
      </Table>
      {footerClassName === undefined || enablePagination ? (
        footerClassName === undefined ? (
          enablePagination ? (
            <DataTablePagination table={table} />
          ) : null
        ) : enablePagination && footerClassName !== '' ? (
          <div className={cn('border-t border-border-soft', footerClassName)}>
            <DataTablePagination table={table} />
          </div>
        ) : null
      ) : null}
    </div>
  )
}

export type DataTableSortingState = SortingState
export type DataTableColumnVisibility = ColumnVisibilityState
export type DataTableFlexRenderProps = ComponentPropsWithoutRef<'th'> & { colSpan?: number }
export type { ColumnDef, DataTableColumn, TanStackTable } from '@/components/ui/data-table-features'
