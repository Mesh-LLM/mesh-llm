import { expect, test, type Page } from '../fixtures/base'

const BREAKPOINT_WIDTHS = [360, 420, 500, 640, 767, 768, 800, 840, 900, 1023, 1024, 1133, 1280, 1440]

async function readTopNavMetrics(page: Page) {
  return page.evaluate(() => {
    const header = document.querySelector('header')
    const apiTarget = document.querySelector('[aria-label="API target instructions"]')
    const actionsMenu = document.querySelector('[aria-label="Open navigation actions"]')
    const compactNavigation = document.querySelector('[aria-label="Primary compact navigation"]')
    const joinButton = document.querySelector('[aria-label="Mesh join and invite instructions"]')
    const themeButton = document.querySelector('[aria-label^="Theme:"]')
    const preferencesButton = document.querySelector('[aria-label="Open interface preferences"]')
    const fullTabLabels = [...document.querySelectorAll('a')].filter((element) =>
      ['Network', 'Chat', 'Configuration'].includes((element.textContent ?? '').trim())
    )

    const isVisible = (element: Element | null): element is Element => {
      if (!element) return false
      const style = window.getComputedStyle(element)
      const rect = element.getBoundingClientRect()

      return style.display !== 'none' && style.visibility !== 'hidden' && rect.width > 0 && rect.height > 0
    }

    const visibleControls = [
      apiTarget,
      actionsMenu,
      compactNavigation,
      joinButton,
      themeButton,
      preferencesButton,
      ...fullTabLabels
    ].filter(isVisible)
    const controlTops = visibleControls.map((element) => Math.round(element.getBoundingClientRect().top))
    const controlTopSpread = controlTops.length > 0 ? Math.max(...controlTops) - Math.min(...controlTops) : 0

    return {
      actionsMenuVisible: isVisible(actionsMenu),
      apiTargetVisible: isVisible(apiTarget),
      compactNavigationVisible: isVisible(compactNavigation),
      controlTopSpread,
      fullTabLabels: fullTabLabels.filter(isVisible).map((element) => (element.textContent ?? '').trim()),
      headerHeight: header ? Math.round(header.getBoundingClientRect().height) : 0,
      horizontalOverflow: document.documentElement.scrollWidth > window.innerWidth,
      joinButtonVisible: isVisible(joinButton),
      preferencesButtonVisible: isVisible(preferencesButton),
      themeButtonVisible: isVisible(themeButton)
    }
  })
}

test('top navigation stays on one row at every responsive breakpoint', async ({ page }) => {
  await page.goto('/')
  await expect(page.getByRole('heading', { name: 'Your private mesh' })).toBeVisible()

  for (const width of BREAKPOINT_WIDTHS) {
    await page.setViewportSize({ width, height: 1133 })
    await expect
      .poll(async () => {
        const metrics = await readTopNavMetrics(page)
        let responsiveControlsReady: boolean
        if (width < 1024) {
          responsiveControlsReady =
            metrics.compactNavigationVisible &&
            metrics.fullTabLabels.length === 0 &&
            !metrics.apiTargetVisible &&
            metrics.actionsMenuVisible
        } else {
          responsiveControlsReady =
            !metrics.compactNavigationVisible &&
            metrics.fullTabLabels.join('|') === 'Network|Chat|Configuration' &&
            metrics.apiTargetVisible &&
            !metrics.actionsMenuVisible &&
            metrics.joinButtonVisible &&
            metrics.themeButtonVisible
        }

        return (
          metrics.headerHeight <= 60 &&
          metrics.controlTopSpread <= 3 &&
          !metrics.horizontalOverflow &&
          responsiveControlsReady
        )
      })
      .toBe(true)

    const metrics = await readTopNavMetrics(page)
    expect(metrics.headerHeight, `${width}px header should remain a single row`).toBeLessThanOrEqual(60)
    expect(metrics.controlTopSpread, `${width}px controls should not split onto separate rows`).toBeLessThanOrEqual(3)
    expect(metrics.horizontalOverflow, `${width}px should not create horizontal document overflow`).toBe(false)

    if (width < 1024) {
      expect(metrics.compactNavigationVisible, `${width}px compact state shows primary icon navigation`).toBe(true)
      expect(metrics.fullTabLabels, `${width}px compact state hides full primary labels`).toEqual([])
      expect(metrics.apiTargetVisible, `${width}px compact state hides API target chip`).toBe(false)
      expect(metrics.actionsMenuVisible, `${width}px compact state keeps actions in the menu`).toBe(true)
    } else {
      expect(metrics.compactNavigationVisible, `${width}px desktop state hides primary icon navigation`).toBe(false)
      expect(metrics.fullTabLabels, `${width}px desktop state shows full primary labels`).toEqual([
        'Network',
        'Chat',
        'Configuration'
      ])
      expect(metrics.apiTargetVisible, `${width}px shows the API target chip`).toBe(true)
      expect(metrics.actionsMenuVisible, `${width}px desktop state shows direct actions`).toBe(false)
      expect(metrics.joinButtonVisible, `${width}px desktop state shows join actions`).toBe(true)
      expect(metrics.themeButtonVisible, `${width}px desktop state shows theme controls`).toBe(true)
    }
  }
})
