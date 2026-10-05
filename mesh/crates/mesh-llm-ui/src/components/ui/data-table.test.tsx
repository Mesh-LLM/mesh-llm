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

    await user.click(screen.getByRole('button', { name: /Go to next page/ }))
    await user.click(screen.getByRole('button', { name: /Name/i }))
    await user.click(await screen.findByRole('menuitem', { name: 'Asc' }))

    const settledRenders = renders
    vi.useFakeTimers()
    await act(async () => vi.advanceTimersByTime(100))
    expect(renders).toBe(settledRenders)
    expect(renders).toBeLessThan(20)
    expect(screen.getByRole('button', { name: 'Name, sorted ascending' })).toBeInTheDocument()
    expect(screen.getByText('row-10')).toBeInTheDocument()
  })

  it('settles after a page change instead of re-rendering in a loop', async () => {
    const user = userEvent.setup()
    const { rerender } = render(<DataTable columns={columns} data={rows} enablePagination />)

    await user.click(screen.getByRole('button', { name: /Go to next page/ }))
    expect(screen.getByText('row-10')).toBeInTheDocument()
    expect(screen.queryByText('row-0')).not.toBeInTheDocument()

    const refreshedRows = rows.map((row) => ({ ...row, name: `${row.name}-refreshed` }))
    rerender(<DataTable columns={columns} data={refreshedRows} enablePagination />)

    expect(screen.getByText('row-10-refreshed')).toBeInTheDocument()
    expect(screen.queryByText('row-0-refreshed')).not.toBeInTheDocument()
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

  it('clamps to the final page when refreshed data shrinks', async () => {
    const user = userEvent.setup()
    const { rerender } = render(<DataTable columns={columns} data={rows} enablePagination />)

    await user.click(screen.getByRole('button', { name: /Go to next page/ }))
    await user.click(screen.getByRole('button', { name: /Go to next page/ }))
    expect(screen.getByText('row-20')).toBeInTheDocument()

    rerender(<DataTable columns={columns} data={rows.slice(0, 15)} enablePagination />)

    expect(screen.getByText('row-10')).toBeInTheDocument()
    expect(screen.queryByText('row-0')).not.toBeInTheDocument()
  })

  it('never commits an impossible page while refreshed data shrinks', async () => {
    const user = userEvent.setup()
    const snapshots: Array<{ pageIndex: number; pageCount: number; rowCount: number }> = []
    const { rerender } = render(
      <DataTable columns={columns} data={rows} enablePagination>
        {(table) => {
          snapshots.push({
            pageIndex: table.state.pagination.pageIndex,
            pageCount: table.getPageCount(),
            rowCount: table.getRowModel().rows.length
          })
          return null
        }}
      </DataTable>
    )

    await user.click(screen.getByRole('button', { name: /Go to next page/ }))
    await user.click(screen.getByRole('button', { name: /Go to next page/ }))
    rerender(
      <DataTable columns={columns} data={rows.slice(0, 15)} enablePagination>
        {(table) => {
          snapshots.push({
            pageIndex: table.state.pagination.pageIndex,
            pageCount: table.getPageCount(),
            rowCount: table.getRowModel().rows.length
          })
          return null
        }}
      </DataTable>
    )

    expect(snapshots).not.toContainEqual({ pageIndex: 2, pageCount: 2, rowCount: 0 })
    expect(snapshots.at(-1)).toEqual({ pageIndex: 1, pageCount: 2, rowCount: 5 })
  })

  it('clamps to the final page when filtering reduces the page count', async () => {
    const user = userEvent.setup()
    render(<DataTable columns={columns} data={rows} enablePagination filterColumnId="name" />)

    await user.click(screen.getByRole('button', { name: /Go to next page/ }))
    await user.click(screen.getByRole('button', { name: /Go to next page/ }))
    expect(screen.getByText('row-20')).toBeInTheDocument()

    await user.type(screen.getByLabelText('Filter...'), 'row-1')

    expect(screen.getByText('row-19')).toBeInTheDocument()
    expect(screen.queryByText('row-0')).not.toBeInTheDocument()
  })

  it('never commits an impossible page while filtering reduces the page count', async () => {
    const user = userEvent.setup()
    const snapshots: Array<{ pageIndex: number; pageCount: number; rowCount: number }> = []
    render(
      <DataTable columns={columns} data={rows} enablePagination filterColumnId="name">
        {(table) => {
          snapshots.push({
            pageIndex: table.state.pagination.pageIndex,
            pageCount: table.getPageCount(),
            rowCount: table.getRowModel().rows.length
          })
          return null
        }}
      </DataTable>
    )

    await user.click(screen.getByRole('button', { name: /Go to next page/ }))
    await user.click(screen.getByRole('button', { name: /Go to next page/ }))
    await user.type(screen.getByLabelText('Filter...'), 'row-1')

    expect(snapshots).not.toContainEqual({ pageIndex: 2, pageCount: 2, rowCount: 0 })
    expect(snapshots.at(-1)).toEqual({ pageIndex: 1, pageCount: 2, rowCount: 1 })
  })

  it('renders the valid empty page immediately when all data is removed', async () => {
    const user = userEvent.setup()
    const snapshots: Array<{ pageIndex: number; pageCount: number; rowCount: number }> = []
    const { rerender } = render(
      <DataTable columns={columns} data={rows} enablePagination>
        {(table) => {
          snapshots.push({
            pageIndex: table.state.pagination.pageIndex,
            pageCount: table.getPageCount(),
            rowCount: table.getRowModel().rows.length
          })
          return null
        }}
      </DataTable>
    )

    await user.click(screen.getByRole('button', { name: /Go to next page/ }))
    await user.click(screen.getByRole('button', { name: /Go to next page/ }))
    rerender(
      <DataTable columns={columns} data={[]} enablePagination>
        {(table) => {
          snapshots.push({
            pageIndex: table.state.pagination.pageIndex,
            pageCount: table.getPageCount(),
            rowCount: table.getRowModel().rows.length
          })
          return null
        }}
      </DataTable>
    )

    expect(snapshots.at(-1)).toEqual({ pageIndex: 0, pageCount: 0, rowCount: 0 })
    expect(screen.getByText('Page 0 of 0')).toBeInTheDocument()
    expect(screen.getByText('No results.')).toBeInTheDocument()
  })

  it('clamps the current page when a larger page size reduces page count', async () => {
    const user = userEvent.setup()
    render(<DataTable columns={columns} data={rows} enablePagination />)

    await user.click(screen.getByRole('button', { name: /Go to next page/ }))
    await user.click(screen.getByRole('button', { name: /Go to next page/ }))
    await user.selectOptions(screen.getByRole('combobox', { name: 'Rows per page' }), '25')

    expect(screen.getByText('Page 1 of 1')).toBeInTheDocument()
    expect(screen.getByText('row-0')).toBeInTheDocument()
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
