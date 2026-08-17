import { act, render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { DataTable, type ColumnDef, type DataTableProps } from '@/components/ui/data-table'
import { DataTableColumnHeader } from '@/components/ui/data-table-column-header'
import { DataTableViewOptions } from '@/components/ui/data-table-view-options'

type Row = { id: string; name: string }

const rows: Row[] = Array.from({ length: 25 }, (_, index) => ({ id: `r${index}`, name: `row-${index}` }))

const columns: ColumnDef<Row, unknown>[] = [
  { accessorKey: 'id', header: 'ID' },
  {
    accessorKey: 'name',
    header: ({ column }) => <DataTableColumnHeader column={column} title="Name" />
  }
]

afterEach(() => vi.useRealTimers())

describe('DataTable', () => {
  it('settles after a sort instead of re-rendering in a loop', async () => {
    const user = userEvent.setup()
    let renders = 0
    function TrackedDataTable(props: DataTableProps<Row>) {
      renders += 1
      return <DataTable {...props} />
    }
    render(<TrackedDataTable columns={columns} data={rows} enablePagination />)

    await user.click(screen.getByRole('button', { name: /Name/i }))
    await user.click(await screen.findByRole('menuitem', { name: 'Asc' }))

    const settledRenders = renders
    vi.useFakeTimers()
    await act(async () => vi.advanceTimersByTime(100))
    expect(renders).toBe(settledRenders)
    expect(renders).toBeLessThan(20)
    expect(screen.getByRole('button', { name: 'Name, sorted ascending' })).toBeInTheDocument()
  })

  it('settles after a page change instead of re-rendering in a loop', async () => {
    const user = userEvent.setup()
    render(<DataTable columns={columns} data={rows} enablePagination />)

    await user.click(screen.getByRole('button', { name: /Go to next page/ }))
    expect(screen.getByText('row-10')).toBeInTheDocument()
    expect(screen.queryByText('row-0')).not.toBeInTheDocument()
  })

  it('settles while typing a filter instead of re-rendering in a loop', async () => {
    const user = userEvent.setup()
    let renders = 0
    function TrackedDataTable(props: DataTableProps<Row>) {
      renders += 1
      return <DataTable {...props} />
    }
    render(<TrackedDataTable columns={columns} data={rows} enablePagination filterColumnId="name" />)

    await user.type(screen.getByLabelText('Filter...'), 'row-1')

    const settledRenders = renders
    vi.useFakeTimers()
    await act(async () => vi.advanceTimersByTime(100))
    expect(renders).toBe(settledRenders)
    expect(renders).toBeLessThan(20)
  })

  it('reflects column visibility changes when the Columns menu is reopened', async () => {
    const user = userEvent.setup()
    render(
      <DataTable columns={columns} data={rows} enablePagination>
        {(table) => <DataTableViewOptions table={table} />}
      </DataTable>
    )

    const openColumns = async () => {
      await user.click(screen.getByRole('button', { name: /columns/i }))
      return screen.findByRole('menuitemcheckbox', { name: 'name' })
    }

    const nameItem = await openColumns()
    expect(nameItem).toHaveAttribute('aria-checked', 'true')

    await user.click(nameItem)
    const reopened = await openColumns()
    expect(reopened).toHaveAttribute('aria-checked', 'false')

    await user.click(reopened)
    const reopenedAgain = await openColumns()
    expect(reopenedAgain).toHaveAttribute('aria-checked', 'true')
  })

  it('uses human-readable labels in the Columns menu when provided', async () => {
    const user = userEvent.setup()
    render(
      <DataTable columns={columns} data={rows} enablePagination>
        {(table) => <DataTableViewOptions columnLabels={{ id: 'Identifier', name: 'Display name' }} table={table} />}
      </DataTable>
    )

    await user.click(screen.getByRole('button', { name: /columns/i }))

    expect(screen.getByRole('menuitemcheckbox', { name: 'Identifier' })).toBeVisible()
    expect(screen.getByRole('menuitemcheckbox', { name: 'Display name' })).toBeVisible()
    expect(screen.queryByRole('menuitemcheckbox', { name: 'name' })).not.toBeInTheDocument()
  })

  it('uses visible columns for the empty span after a column is hidden', async () => {
    const user = userEvent.setup()
    render(
      <DataTable columns={columns} data={[]} enablePagination>
        {(table) => <DataTableViewOptions table={table} />}
      </DataTable>
    )
    const emptyCell = screen.getByText('No results.').closest('td')
    expect(emptyCell).toHaveAttribute('colspan', '2')
    expect(screen.getByText('Page 0 of 0')).toBeInTheDocument()

    await user.click(screen.getByRole('button', { name: /columns/i }))
    await user.click(screen.getByRole('menuitemcheckbox', { name: 'name' }))

    expect(emptyCell).toHaveAttribute('colspan', '1')
  })

  it('activates a labeled row with pointer and keyboard input', async () => {
    const user = userEvent.setup()
    const onRowActivate = vi.fn()
    render(
      <DataTable
        columns={columns}
        data={rows.slice(0, 1)}
        getRowAriaLabel={(row) => `Inspect ${row.name}`}
        onRowActivate={onRowActivate}
      />
    )
    const row = screen.getByRole('row', { name: 'Inspect row-0' })

    await user.click(row)
    row.focus()
    await user.keyboard('{Enter}')

    expect(onRowActivate).toHaveBeenNthCalledWith(1, rows[0])
    expect(onRowActivate).toHaveBeenNthCalledWith(2, rows[0])
  })
})
