import type { RowData } from '@tanstack/react-table'
import { Columns3 } from 'lucide-react'
import { Button } from '@/components/ui/button'
import type { TanStackTable } from '@/components/ui/data-table-features'
import {
  DropdownMenu,
  DropdownMenuCheckboxItem,
  DropdownMenuContent,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger
} from '@/components/ui/DropdownMenu'

type DataTableViewOptionsProps<TData extends RowData> = {
  readonly table: TanStackTable<TData>
  readonly columnLabels?: Readonly<Record<string, string>>
}

export function DataTableViewOptions<TData extends RowData>({ table, columnLabels }: DataTableViewOptionsProps<TData>) {
  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <Button
          className="ui-control h-8 gap-1.5 rounded-[var(--radius)] px-2.5 text-[length:var(--density-type-caption)]"
          size="sm"
          variant="outline"
        >
          <Columns3 className="size-3.5" aria-hidden="true" />
          Columns
        </Button>
      </DropdownMenuTrigger>
      <DropdownMenuContent align="end">
        <DropdownMenuLabel>Toggle columns</DropdownMenuLabel>
        <DropdownMenuSeparator />
        {table
          .getAllColumns()
          .filter((column) => typeof column.accessorFn !== 'undefined' && column.getCanHide())
          .map((column) => (
            <DropdownMenuCheckboxItem
              checked={column.getIsVisible()}
              key={column.id}
              onCheckedChange={(value) => column.toggleVisibility(!!value)}
            >
              {columnLabels?.[column.id] ?? column.id}
            </DropdownMenuCheckboxItem>
          ))}
      </DropdownMenuContent>
    </DropdownMenu>
  )
}
