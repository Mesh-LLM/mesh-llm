import { openInspector, selectInspectorTab } from './request-inspector-helpers'
import {
  installRequestInspectorRoutes,
  REQUEST_INSPECTOR_ARTIFACT_IDS,
  REQUEST_INSPECTOR_IDS
} from './request-inspector-routes'
import { expect, test } from './request-inspector-test'

test('loads only the selected payload and safely exposes format, lines, and copy controls', async ({
  context,
  page
}) => {
  await context.grantPermissions(['clipboard-read', 'clipboard-write'])
  const backend = await installRequestInspectorRoutes(page)
  await page.goto('/logs')
  const inspector = await openInspector(page, REQUEST_INSPECTOR_IDS.completed)

  expect(backend.artifactDetailCalls).toEqual([])
  await selectInspectorTab(page, inspector, { id: 'payloads', name: 'Payloads' })
  const requestPane = inspector.getByRole('region', { name: 'Request', exact: true })
  const responsePane = inspector.getByRole('region', { name: 'Response', exact: true })

  // Only the Request pane is visible by default (single-pane toggle)
  await expect(inspector.getByRole('button', { name: 'Load payload' })).toHaveCount(1)
  for (const state of ['missing', 'unavailable', 'corrupt'] as const) {
    await expect(inspector.getByText(state, { exact: true }).first()).toBeVisible()
  }
  expect(backend.artifactDetailCalls).toEqual([])

  // Load request payload
  await requestPane.getByRole('button', { name: 'Load payload' }).click()
  await expect.poll(() => backend.artifactDetailCalls).toEqual([REQUEST_INSPECTOR_ARTIFACT_IDS.request])
  const requestJson = requestPane.getByRole('region', { name: 'Request JSON payload' })
  await expect(requestJson.locator('[data-json-token="key"]').first()).toBeVisible()
  await expect(requestJson.locator('[data-line-number]')).toHaveCount(5)
  await expect(requestJson.getByRole('radio', { name: 'Pretty' })).toBeChecked()

  // Format, lines, copy controls on request JSON
  await requestJson.getByRole('radio', { name: 'Raw' }).click()
  await expect(requestJson.getByRole('radio', { name: 'Raw' })).toBeChecked()
  await expect(requestJson.locator('[data-line-number]')).toHaveCount(1)
  const copy = requestJson.getByRole('button', { name: 'Copy JSON payload' })
  await expect(copy).toBeVisible()
  await copy.click()
  await expect(requestJson.getByRole('status')).toContainText('Raw JSON representation selected. JSON payload copied.')
  await expect(requestJson.getByText(/<img src=payload onerror=alert\(2\)>/)).toBeVisible()
  await expect(requestJson.getByText(/<script>globalThis\.compromised=true<\/script>/)).toBeVisible()
  await expect(requestJson.locator('img')).toHaveCount(0)
  await expect(requestJson.locator('script')).toHaveCount(0)

  // Toggle to Response, load response payload
  await inspector.getByRole('radio', { name: 'Response' }).click()
  await responsePane.getByRole('button', { name: 'Load payload' }).click()
  await expect
    .poll(() => backend.artifactDetailCalls)
    .toEqual([REQUEST_INSPECTOR_ARTIFACT_IDS.request, REQUEST_INSPECTOR_ARTIFACT_IDS.response])
  const responseJson = responsePane.getByRole('region', { name: 'Response JSON payload' })
  await expect(responseJson.locator('[data-json-token="boolean"]').first()).toBeVisible()

  // Tab round-trip — both panes cached, no new artifactDetailCalls
  // Toggle resets to Request on remount after tab switch
  await selectInspectorTab(page, inspector, { id: 'overview', name: 'Overview' })
  await selectInspectorTab(page, inspector, { id: 'payloads', name: 'Payloads' })

  // Request pane shows cached ready-to-view state (1 View payload button visible)
  await expect(requestPane.getByRole('button', { name: 'View payload' })).toBeVisible()
  await requestPane.getByRole('button', { name: 'View payload' }).click()
  await expect(requestJson).toBeVisible()

  // Toggle to Response — also cached, no new fetches
  await inspector.getByRole('radio', { name: 'Response' }).click()
  await responsePane.getByRole('button', { name: 'View payload' }).click()
  await expect(responseJson).toBeVisible()

  expect(backend.artifactDetailCalls).toEqual([
    REQUEST_INSPECTOR_ARTIFACT_IDS.request,
    REQUEST_INSPECTOR_ARTIFACT_IDS.response
  ])
})

test('renders malformed retained JSON as inert plaintext only after explicit load', async ({ page }) => {
  const backend = await installRequestInspectorRoutes(page)
  await page.goto('/logs')
  const inspector = await openInspector(page, REQUEST_INSPECTOR_IDS.malformed)
  await selectInspectorTab(page, inspector, { id: 'payloads', name: 'Payloads' })

  expect(backend.artifactDetailCalls).toEqual([])
  await inspector
    .getByRole('region', { name: 'Request', exact: true })
    .getByRole('button', {
      name: 'Load payload'
    })
    .click()
  await expect.poll(() => backend.artifactDetailCalls).toEqual([REQUEST_INSPECTOR_ARTIFACT_IDS.malformed])
  await expect(inspector.getByText('Malformed JSON. Showing inert plaintext; no markup is interpreted.')).toBeVisible()
  await expect(inspector.getByRole('region', { name: 'Request malformed JSON plaintext' })).toContainText(
    '<img src=malformed onerror=alert(3)>'
  )
  await expect(inspector.getByRole('button', { name: 'Copy JSON payload' })).toHaveCount(0)
  await expect(inspector.locator('img')).toHaveCount(0)
  await expect(inspector.locator('script')).toHaveCount(0)
})
