import type { Locator, Page } from '@playwright/test'
import { expect } from './request-inspector-test'

export type RequestInspectorTab = {
  readonly id: 'overview' | 'payloads' | 'timeline' | 'diagnostics'
  readonly name: 'Overview' | 'Payloads' | 'Timeline' | 'Diagnostics'
}

export function requestRow(page: Page, requestId: string): Locator {
  return page.getByRole('row', { name: `Inspect request ${requestId}` })
}

export async function openInspector(page: Page, requestId: string): Promise<Locator> {
  const row = requestRow(page, requestId)
  await expect(row).toBeVisible()
  await row.click()
  const inspector = page.getByRole('dialog', { name: 'Request Inspector' })
  await expect(inspector).toBeVisible()
  await expect(page).toHaveURL(new RegExp(`inspectType=request&inspectId=${requestId}`))
  return inspector
}

export async function selectInspectorTab(page: Page, inspector: Locator, tab: RequestInspectorTab): Promise<void> {
  const trigger = inspector.getByRole('tab', { name: tab.name, exact: true })
  await trigger.click()
  await expect(trigger).toHaveAttribute('data-state', 'active')
  await expect.poll(() => new URL(page.url()).searchParams.get('tab')).toBe(tab.id)
}
