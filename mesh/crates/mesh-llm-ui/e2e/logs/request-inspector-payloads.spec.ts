import { openInspector, selectInspectorTab } from './request-inspector-helpers'
import {
  installRequestInspectorRoutes,
  REQUEST_INSPECTOR_ARTIFACT_IDS,
  REQUEST_INSPECTOR_IDS,
  REQUEST_INSPECTOR_STREAM_HOSTILE_TEXT
} from './request-inspector-routes'
import { expect, test } from './request-inspector-test'
import type { Locator, Page } from '@playwright/test'

type ScrollEndpoint = 'start' | 'end'

async function expectCopyAnchoredAtEndpoint(
  page: Page,
  payloadViewport: Locator,
  copy: Locator,
  endpoint: ScrollEndpoint
) {
  const expectedState = {
    atEndpoint: true,
    copyContained: true,
    copyPainted: true,
    horizontallyScrollable: true,
    toolbarMatchesViewport: true
  }

  await payloadViewport.evaluate((viewport, target) => {
    viewport.scrollLeft = target === 'start' ? 0 : viewport.scrollWidth - viewport.clientWidth
  }, endpoint)
  await expect
    .poll(() =>
      payloadViewport.evaluate((viewport, target) => {
        const button = viewport.querySelector<HTMLElement>('button[aria-label="Copy JSON payload"]')
        const buttonRect = button?.getBoundingClientRect()
        const toolbarRect = button?.parentElement?.getBoundingClientRect()
        const viewportRect = viewport.getBoundingClientRect()
        const targetScrollLeft = target === 'start' ? 0 : viewport.scrollWidth - viewport.clientWidth
        const hitTarget =
          buttonRect === undefined
            ? null
            : document.elementFromPoint(buttonRect.left + buttonRect.width / 2, buttonRect.top + buttonRect.height / 2)

        return {
          atEndpoint: Math.abs(viewport.scrollLeft - targetScrollLeft) <= 1,
          copyContained:
            buttonRect !== undefined &&
            buttonRect.left >= viewportRect.left - 1 &&
            buttonRect.right <= viewportRect.right + 1,
          copyPainted: button !== null && hitTarget !== null && (hitTarget === button || button.contains(hitTarget)),
          horizontallyScrollable: viewport.scrollWidth > viewport.clientWidth,
          toolbarMatchesViewport: toolbarRect !== undefined && Math.abs(toolbarRect.width - viewportRect.width) <= 1
        }
      }, endpoint)
    )
    .toEqual(expectedState)

  await payloadViewport.focus()
  await page.keyboard.press('Tab')
  await expect(copy).toBeFocused()
  await expect
    .poll(() =>
      payloadViewport.evaluate((viewport, target) => {
        const targetScrollLeft = target === 'start' ? 0 : viewport.scrollWidth - viewport.clientWidth
        return Math.abs(viewport.scrollLeft - targetScrollLeft) <= 1
      }, endpoint)
    )
    .toBe(true)
}

test('loads only the selected payload and safely exposes format, lines, and copy controls', async ({
  context,
  page
}) => {
  await page.setViewportSize({ width: 375, height: 812 })
  await context.grantPermissions(['clipboard-read', 'clipboard-write'])
  const backend = await installRequestInspectorRoutes(page)
  await page.goto('/logs')
  const inspector = await openInspector(page, REQUEST_INSPECTOR_IDS.completed)

  expect(backend.artifactDetailCalls).toEqual([])
  await selectInspectorTab(page, inspector, { id: 'payloads', name: 'Payloads' })
  const requestPane = inspector.getByRole('region', { name: 'Request', exact: true })
  const responsePane = inspector.getByRole('region', { name: 'Response', exact: true })

  // Only the Request pane is visible by default (single-pane toggle)
  for (const state of ['missing', 'unavailable', 'corrupt'] as const) {
    await expect(inspector.getByText(state, { exact: true }).first()).toBeVisible()
  }
  await expect.poll(() => backend.artifactDetailCalls).toEqual([REQUEST_INSPECTOR_ARTIFACT_IDS.request])
  const requestJson = requestPane.getByRole('region', { name: 'Request JSON payload' })
  const paneHeader = requestPane.locator('header').first()
  const payloadControl = paneHeader.getByRole('radiogroup', { name: 'Payload' })
  const displayToolbar = inspector.getByRole('toolbar', { name: 'Display' })
  const formatControl = displayToolbar.getByRole('radiogroup', { name: 'Display' })
  await expect(payloadControl.getByRole('radio', { name: 'Request' })).toBeChecked()
  await expect(payloadControl.getByRole('radio', { name: 'Response' })).toBeVisible()
  await expect(displayToolbar).toHaveCount(1)
  await expect(formatControl).toHaveCount(1)
  await expect(paneHeader.getByRole('radio', { name: 'Pretty' })).toHaveCount(0)
  await expect(displayToolbar.getByRole('radio', { name: 'Request' })).toHaveCount(0)
  await expect(requestJson.getByRole('radiogroup', { name: 'Display' })).toHaveCount(0)
  await expect(requestJson.locator('[data-json-token="key"]').first()).toBeVisible()
  await expect(requestJson.locator('[data-line-number]')).toHaveCount(5)
  await expect(formatControl.getByRole('radio', { name: 'Pretty' })).toBeChecked()

  const payloadViewport = requestPane.getByRole('region', { name: 'Request payload content' })
  const copy = requestJson.getByRole('button', { name: 'Copy JSON payload' })
  const scrollArea = payloadViewport.locator('..')
  const horizontalScrollbar = scrollArea.locator('[data-orientation="horizontal"]')
  const verticalScrollbar = scrollArea.locator('[data-orientation="vertical"]')
  const horizontalThumb = horizontalScrollbar.locator(':scope > div').first()
  const inspectorBody = inspector.locator('[data-request-inspector-scroll="body"]')
  await expect(inspectorBody).toHaveCount(1)
  await expect
    .poll(() =>
      inspectorBody.evaluate((body) => ({
        overflowY: getComputedStyle(body).overflowY,
        verticallyScrollable: body.scrollHeight > body.clientHeight
      }))
    )
    .toEqual({ overflowY: 'auto', verticallyScrollable: true })
  await expect(inspector.locator('[data-radix-scroll-area-viewport]')).toHaveCount(1)
  await expect(payloadViewport.locator('[data-radix-scroll-area-viewport]')).toHaveCount(0)
  await expect(horizontalScrollbar).toBeVisible()
  await expect(horizontalThumb).toBeVisible()
  await expect(verticalScrollbar).toHaveCount(0)
  await expect(scrollArea).not.toHaveClass(/(?:^|\s)(?:h-64|h-80|sm:h-\[28rem\]|lg:h-\[32rem\])(?:\s|$)/)
  await expect
    .poll(() =>
      payloadViewport.evaluate((viewport) => ({
        overflowY: getComputedStyle(viewport).overflowY,
        verticallyScrollable: viewport.scrollHeight > viewport.clientHeight
      }))
    )
    .toEqual({ overflowY: 'hidden', verticallyScrollable: false })
  const prettyPayloadHeight = await payloadViewport.evaluate((viewport) => viewport.getBoundingClientRect().height)
  await expectCopyAnchoredAtEndpoint(page, payloadViewport, copy, 'start')
  await expectCopyAnchoredAtEndpoint(page, payloadViewport, copy, 'end')
  await expect
    .poll(() =>
      horizontalScrollbar.evaluate((track) => {
        const thumb = track.firstElementChild
        const trackColor = getComputedStyle(track).backgroundColor
        const thumbColor = thumb === null ? '' : getComputedStyle(thumb).backgroundColor
        return {
          colorsDiffer: trackColor !== thumbColor,
          thumbOpaque: thumbColor !== '' && thumbColor !== 'rgba(0, 0, 0, 0)',
          trackOpaque: trackColor !== 'rgba(0, 0, 0, 0)'
        }
      })
    )
    .toEqual({ colorsDiffer: true, thumbOpaque: true, trackOpaque: true })

  // Format, lines, copy controls on request JSON
  await formatControl.getByRole('radio', { name: 'Raw' }).click()
  await expect(formatControl.getByRole('radio', { name: 'Raw' })).toBeChecked()
  await expect(requestJson.locator('[data-line-number]')).toHaveCount(1)
  await expect
    .poll(() => payloadViewport.evaluate((viewport) => viewport.getBoundingClientRect().height))
    .toBeLessThan(prettyPayloadHeight)
  await expectCopyAnchoredAtEndpoint(page, payloadViewport, copy, 'start')
  await expectCopyAnchoredAtEndpoint(page, payloadViewport, copy, 'end')
  await expect(copy).toBeVisible()
  await copy.click()
  await expect(requestJson.getByRole('status')).toContainText('Raw JSON representation selected. JSON payload copied.')
  await expect(requestJson.getByText(/<img src=payload onerror=alert\(2\)>/)).toBeVisible()
  await expect(requestJson.getByText(/<script>globalThis\.compromised=true<\/script>/)).toBeVisible()
  await expect(requestJson.locator('img')).toHaveCount(0)
  await expect(requestJson.locator('script')).toHaveCount(0)

  await inspector.getByRole('radio', { name: 'Response' }).click()
  await expect
    .poll(() => backend.artifactDetailCalls)
    .toEqual([REQUEST_INSPECTOR_ARTIFACT_IDS.request, REQUEST_INSPECTOR_ARTIFACT_IDS.response])
  const responseJson = responsePane.getByRole('region', { name: 'Response JSON payload' })
  await expect(responseJson.locator('[data-json-token="boolean"]').first()).toBeVisible()
  await expect(formatControl.getByRole('radio', { name: 'Raw' })).toBeChecked()
  await expect(responseJson.locator('[data-line-number]')).toHaveCount(1)

  await inspector.getByRole('radio', { name: 'Request' }).click()
  await expect(formatControl.getByRole('radio', { name: 'Raw' })).toBeChecked()
  await expect(requestJson.locator('[data-line-number]')).toHaveCount(1)

  // Tab round-trip — both panes cached, no new artifactDetailCalls
  await selectInspectorTab(page, inspector, { id: 'overview', name: 'Overview' })
  await selectInspectorTab(page, inspector, { id: 'payloads', name: 'Payloads' })

  await expect(requestJson).toBeVisible()

  await inspector.getByRole('radio', { name: 'Response' }).click()
  await expect(responseJson).toBeVisible()

  expect(backend.artifactDetailCalls).toEqual([
    REQUEST_INSPECTOR_ARTIFACT_IDS.request,
    REQUEST_INSPECTOR_ARTIFACT_IDS.response
  ])
})

test('pages one multi-frame SSE response with numbered controls while preserving format and inert frame content', async ({
  page
}) => {
  await page.setViewportSize({ width: 375, height: 812 })
  const backend = await installRequestInspectorRoutes(page)
  await page.goto('/logs')
  const inspector = await openInspector(page, REQUEST_INSPECTOR_IDS.streaming)
  await selectInspectorTab(page, inspector, { id: 'payloads', name: 'Payloads' })

  const displayToolbar = inspector.getByRole('toolbar', { name: 'Display' })
  const formatControl = displayToolbar.getByRole('radiogroup', { name: 'Display' })
  await expect(displayToolbar).toHaveCount(1)
  await expect(formatControl).toHaveCount(1)
  await formatControl.getByRole('radio', { name: 'Raw' }).click()
  await inspector.getByRole('radio', { name: 'Response' }).click()
  await expect.poll(() => backend.artifactDetailCalls).toEqual([REQUEST_INSPECTOR_ARTIFACT_IDS.streamingResponse])

  const payloadViewport = inspector.getByRole('region', { name: 'Response payload content' })
  const firstFrame = inspector.getByRole('region', { name: 'Response event stream frame 1', exact: true })
  await expect(firstFrame).toContainText('Frame 1 of 3')
  await expect(firstFrame).toContainText('delta')
  await expect(firstFrame).toContainText('stream-1')
  await expect(firstFrame.locator('[data-json-line]')).toHaveCount(1)
  await expect(firstFrame.getByRole('button', { name: 'Copy JSON payload' })).toBeVisible()
  await expect(inspector.getByRole('region', { name: /^Response event stream frame \d$/ })).toHaveCount(1)
  await expect(inspector.getByRole('listitem')).toHaveCount(0)
  await expect(payloadViewport.locator('[data-radix-scroll-area-viewport]')).toHaveCount(0)
  const framePager = inspector.getByRole('radiogroup', { name: 'Response frames' })
  const frameNavigator = framePager.locator('..')
  const frameChoices = framePager.getByRole('radio')
  await expect(frameChoices).toHaveText(['1', '2', '3'])
  const previous = inspector.getByRole('button', { name: 'Previous response frame' })
  const next = inspector.getByRole('button', { name: 'Next response frame' })
  await expect(previous).toBeDisabled()
  await expect(next).toBeEnabled()
  await expect(inspector.getByRole('status').filter({ hasText: 'Frame 1 of 3' })).toHaveCount(1)
  await expect
    .poll(() =>
      frameNavigator.evaluate((navigator) => {
        const header = navigator.closest('header')
        const context = header?.firstElementChild
        if (!(header instanceof HTMLElement) || !(context instanceof HTMLElement)) {
          return { fillsRow: false, followsContext: false }
        }
        const navigatorRect = navigator.getBoundingClientRect()
        const headerRect = header.getBoundingClientRect()
        const contextRect = context.getBoundingClientRect()
        return {
          fillsRow: navigatorRect.width >= headerRect.width - 26,
          followsContext: navigatorRect.top >= contextRect.bottom
        }
      })
    )
    .toEqual({ fillsRow: true, followsContext: true })
  for (const target of [previous, ...(await frameChoices.all()), next]) {
    await expect
      .poll(() =>
        target.evaluate((control) => {
          const rect = control.getBoundingClientRect()
          return Math.min(rect.width, rect.height)
        })
      )
      .toBeGreaterThanOrEqual(32)
  }

  await framePager.getByRole('radio', { name: 'Response frame 2 of 3' }).click()
  const secondFrame = inspector.getByRole('region', { name: 'Response event stream frame 2', exact: true })
  await expect(firstFrame).toHaveCount(0)
  await expect(secondFrame).toContainText(REQUEST_INSPECTOR_STREAM_HOSTILE_TEXT)
  await expect(secondFrame.locator('img')).toHaveCount(0)
  await expect(secondFrame.locator('script')).toHaveCount(0)
  await expect(secondFrame.getByRole('button', { name: 'Copy JSON payload' })).toHaveCount(0)
  await expect(inspector.getByRole('region', { name: /^Response event stream frame \d$/ })).toHaveCount(1)
  await expect(formatControl.getByRole('radio', { name: 'Raw' })).toBeChecked()

  await framePager.getByRole('radio', { name: 'Response frame 2 of 3' }).focus()
  await page.keyboard.press('ArrowRight')
  const doneFrame = inspector.getByRole('region', { name: 'Response event stream frame 3', exact: true })
  await expect(secondFrame).toHaveCount(0)
  await expect(doneFrame).toContainText('Frame 3 of 3')
  await expect(doneFrame).toContainText('done')
  await expect(doneFrame).toContainText('[DONE]')
  await expect(doneFrame.getByRole('button', { name: 'Copy JSON payload' })).toHaveCount(0)
  await expect(inspector.getByRole('region', { name: /^Response event stream frame \d$/ })).toHaveCount(1)
  await expect(next).toBeDisabled()
  await expect(previous).toBeEnabled()
  await expect(inspector.getByRole('status').filter({ hasText: 'Frame 3 of 3' })).toHaveCount(1)
  await expect(formatControl.getByRole('radio', { name: 'Raw' })).toBeChecked()
  expect(backend.artifactDetailCalls).toEqual([REQUEST_INSPECTOR_ARTIFACT_IDS.streamingResponse])
})

test('renders malformed retained JSON as inert plaintext when the selected payload loads', async ({ page }) => {
  const backend = await installRequestInspectorRoutes(page)
  await page.goto('/logs')
  const inspector = await openInspector(page, REQUEST_INSPECTOR_IDS.malformed)
  await selectInspectorTab(page, inspector, { id: 'payloads', name: 'Payloads' })

  await expect.poll(() => backend.artifactDetailCalls).toEqual([REQUEST_INSPECTOR_ARTIFACT_IDS.malformed])
  await expect(inspector.getByText('Malformed JSON. Showing inert plaintext; no markup is interpreted.')).toBeVisible()
  await expect(inspector.getByRole('region', { name: 'Request malformed JSON plaintext' })).toContainText(
    '<img src=malformed onerror=alert(3)>'
  )
  await expect(inspector.getByRole('button', { name: 'Copy JSON payload' })).toHaveCount(0)
  await expect(inspector.locator('img')).toHaveCount(0)
  await expect(inspector.locator('script')).toHaveCount(0)
})
