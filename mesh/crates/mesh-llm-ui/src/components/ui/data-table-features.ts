import {
  type Column as TanStackColumnType,
  type ColumnDef as TanStackColumnDef,
  type ReactTable as TanStackReactTableType,
  type RowData,
  columnFilteringFeature,
  columnVisibilityFeature,
  createFilteredRowModel,
  createPaginatedRowModel,
  createSortedRowModel,
  filterFn_includesString,
  rowPaginationFeature,
  rowSelectionFeature,
  rowSortingFeature,
  sortFn_alphanumeric,
  sortFn_datetime,
  sortFn_text,
  tableFeatures
} from '@tanstack/react-table'

export const dataTableFeatures = tableFeatures({
  columnFilteringFeature,
  filteredRowModel: createFilteredRowModel(),
  filterFns: { includesString: filterFn_includesString },
  columnVisibilityFeature,
  rowPaginationFeature,
  paginatedRowModel: createPaginatedRowModel(),
  rowSelectionFeature,
  rowSortingFeature,
  sortedRowModel: createSortedRowModel(),
  sortFns: {
    alphanumeric: sortFn_alphanumeric,
    datetime: sortFn_datetime,
    text: sortFn_text
  }
})

export type DataTableFeatures = typeof dataTableFeatures
export type ColumnDef<TData extends RowData, TValue = unknown> = TanStackColumnDef<DataTableFeatures, TData, TValue>
export type DataTableColumn<TData extends RowData, TValue = unknown> = TanStackColumnType<
  DataTableFeatures,
  TData,
  TValue
>
export type TanStackTable<TData extends RowData> = TanStackReactTableType<DataTableFeatures, TData>
