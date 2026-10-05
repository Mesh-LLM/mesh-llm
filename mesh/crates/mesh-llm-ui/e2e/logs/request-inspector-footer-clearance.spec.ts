import type { Page } from '@playwright/test'
import { openInspector } from './request-inspector-helpers'
import { installRequestInspectorRoutes, REQUEST_INSPECTOR_IDS } from './request-inspector-routes'
import { expect, test } from './request-inspector-test'

const VIEWPORTS = [
  { label: 'mobile', width: 375, height: 520 },
  { label: 'desktop', width: 1280, height: 800 }
] as const

async function footerClearance(page: Page) {
  const inspector = await openInspector(page, REQUEST_INSPECTOR_IDS.completed)
  const scrollBody = inspector.locator('[data-request-inspector-scroll="body"]')
  const footer = inspector.getByRole('contentinfo', { name: 'Request inspector actions' })
  const overview = inspector.getByRole('region', { name: 'Request overview' })
  await expect.poll(() => scrollBody.evaluate((element) => element.scrollHeight > element.clientHeight)).toBe(true)
  await scrollBody.evaluate((element) => {
    element.scrollTop = element.scrollHeight
  })
  await expect
    .poll(() => scrollBody.evaluate((element) => element.scrollTop + element.clientHeight))
    .toBe(await scrollBody.evaluate((element) => element.scrollHeight))

  const [contentBottom, footerTop, spacing] = await Promise.all([
    overview.evaluate((element) => element.lastElementChild?.getBoundingClientRect().bottom ?? null),
    footer.evaluate((element) => element.getBoundingClientRect().top),
    scrollBody.evaluate((element) => ({
      expected: Number.parseFloat(getComputedStyle(document.documentElement).getPropertyValue('--shell-normal')),
      paddingBottom: Number.parseFloat(getComputedStyle(element).paddingBottom)
    }))
  ])
  if (contentBottom === null) throw new Error('Request overview final panel bounds missing')
  return { contentBottom, footerTop, ...spacing }
}

for (const viewport of VIEWPORTS) {
  test(`keeps the final overview panel one normal shell space above the footer at ${viewport.label} width`, async ({
    page
  }) => {
    await installRequestInspectorRoutes(page)
    await page.setViewportSize({ width: viewport.width, height: viewport.height })
    await page.goto('/logs')

    const { contentBottom, expected, footerTop, paddingBottom } = await footerClearance(page)

    expect(paddingBottom).toBeCloseTo(expected, 1)
    expect(footerTop - contentBottom).toBeGreaterThanOrEqual(expected - 1)
  })
}
