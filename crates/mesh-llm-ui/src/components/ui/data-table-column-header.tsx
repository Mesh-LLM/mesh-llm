import type { RowData } from '@tanstack/react-table'
import { ArrowDown, ArrowUp, ChevronsUpDown, EyeOff } from 'lucide-react'
import { cn } from '@/lib/cn'
import { Button } from '@/components/ui/button'
import type { DataTableColumn } from '@/components/ui/data-table-features'
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger
} from '@/components/ui/DropdownMenu'

type DataTableColumnHeaderProps<TData extends RowData, TValue> = {
  readonly column: DataTableColumn<TData, TValue>
  readonly title: string
  readonly className?: string
}

export function DataTableColumnHeader<TData extends RowData, TValue>({
  column,
  title,
  className
}: DataTableColumnHeaderProps<TData, TValue>) {
  if (!column.getCanSort()) {
    return <div className={cn(className)}>{title}</div>
  }
  const sortState = column.getIsSorted()
  const sortLabel = sortState === 'asc' ? 'sorted ascending' : sortState === 'desc' ? 'sorted descending' : 'not sorted'

  return (
    <div className={cn('flex items-center gap-2', className)}>
      <DropdownMenu>
        <DropdownMenuTrigger asChild>
          <Button
            aria-label={`${title}, ${sortLabel}`}
            className="-ml-2 h-8 gap-1 px-2 text-fg-faint hover:text-foreground data-[state=open]:bg-panel-strong"
            size="sm"
            variant="ghost"
          >
            <span>{title}</span>
            {sortState === 'desc' ? (
              <ArrowDown className="size-3.5" aria-hidden="true" />
            ) : sortState === 'asc' ? (
              <ArrowUp className="size-3.5" aria-hidden="true" />
            ) : (
              <ChevronsUpDown className="size-3.5" aria-hidden="true" />
            )}
          </Button>
        </DropdownMenuTrigger>
        <DropdownMenuContent align="start">
          <DropdownMenuItem onClick={() => column.toggleSorting(false)}>
            <ArrowUp className="size-3.5" aria-hidden="true" />
            Asc
          </DropdownMenuItem>
          <DropdownMenuItem onClick={() => column.toggleSorting(true)}>
            <ArrowDown className="size-3.5" aria-hidden="true" />
            Desc
          </DropdownMenuItem>
          <DropdownMenuSeparator />
          <DropdownMenuItem onClick={() => column.toggleVisibility(false)}>
            <EyeOff className="size-3.5" aria-hidden="true" />
            Hide
          </DropdownMenuItem>
        </DropdownMenuContent>
      </DropdownMenu>
    </div>
  )
}
