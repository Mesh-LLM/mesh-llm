import { mkdir, writeFile } from 'node:fs/promises'
import { dirname, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'
import { expect, test, type APIRequestContext, type Page } from '@playwright/test'
import { DATA_MODE_STORAGE_KEY } from '@e2e/support/data-mode'

const pluginName = 'web-ui-exemplar'
const pluginApi = `/api/plugins/${pluginName}`
const evidenceDirectory = resolve(
  process.env.MESH_PLUGIN_EVIDENCE_DIR ??
    resolve(dirname(fileURLToPath(import.meta.url)), '../../../../target/plugin-web-ui-evidence/playwright')
)

type BrowserDiagnostics = {
  consoleErrors: string[]
  pageErrors: string[]
}

type BrowserApiResponse = {
  method: string
  path: string
  status: number
}

const darkUiPreferences = {
  theme: 'dark',
  accent: 'blue',
  density: 'normal',
  panelStyle: 'soft',
  panelStyleOverride: false
}

async function setWebUiEnabled(request: APIRequestContext, enabled: boolean) {
  const response = await request.patch(`${pluginApi}/web-ui/enabled`, { data: { enabled } })
  expect(response.status()).toBe(200)
}

async function setRetentionDays(request: APIRequestContext, retentionDays: number) {
  return request.patch(`${pluginApi}/web-ui/config`, {
    data: { settings: { retention_days: retentionDays } }
  })
}

async function expectWebUiState(request: APIRequestContext, expectedState: string) {
  await expect
    .poll(async () => {
      const response = await request.get(`${pluginApi}/web-ui`)
      if (!response.ok()) return `http-${response.status()}`
      const body = (await response.json()) as { state?: string }
      return body.state
    })
    .toBe(expectedState)
}

function collectBrowserDiagnostics(page: Page): BrowserDiagnostics {
  const diagnostics: BrowserDiagnostics = { consoleErrors: [], pageErrors: [] }
  page.on('console', (message) => {
    if (message.type() === 'error') {
      const location = message.location()
      diagnostics.consoleErrors.push(
        `${message.text()}${location.url ? ` (${location.url}:${location.lineNumber})` : ''}`
      )
    }
  })
  page.on('pageerror', (error) => diagnostics.pageErrors.push(error.stack ?? error.message))
  return diagnostics
}

function collectBrowserApiResponses(page: Page): BrowserApiResponse[] {
  const responses: BrowserApiResponse[] = []
  page.on('response', (response) => {
    const url = new URL(response.url())
    if (!url.pathname.startsWith('/api/')) return
    responses.push({
      method: response.request().method(),
      path: `${url.pathname}${url.search}`,
      status: response.status()
    })
  })
  return responses
}

async function expectBrowserApiResponse(responses: BrowserApiResponse[], path: string, status = 200) {
  await expect.poll(() => responses.some((response) => response.path === path && response.status === status)).toBe(true)
}

test.describe('installed plugin web UI exemplar @plugin', () => {
  test.skip(!process.env.MESH_PLUGIN_E2E, 'requires the installed live exemplar started by its documented recipe')
  test.describe.configure({ mode: 'serial' })
  test.use({ colorScheme: 'dark', viewport: { width: 1440, height: 1000 } })

  test('is installed, running, rendered, configurable, and independently disableable', async ({ page, request }) => {
    test.slow()
    const diagnostics = collectBrowserDiagnostics(page)
    const browserApiResponses = collectBrowserApiResponses(page)
    await mkdir(evidenceDirectory, { recursive: true })

    await page.addInitScript(
      ({ dataModeStorageKey, preferences }) => {
        window.localStorage.setItem('mesh-llm-ui-preview:preferences:v1', JSON.stringify(preferences))
        window.localStorage.setItem(dataModeStorageKey, 'live')
      },
      { dataModeStorageKey: DATA_MODE_STORAGE_KEY, preferences: darkUiPreferences }
    )

    try {
      const consoleResponse = await request.get('/')
      expect(consoleResponse.status()).toBe(200)
      const consoleHtml = await consoleResponse.text()
      expect(consoleHtml).not.toContain('/@vite/client')

      await setWebUiEnabled(request, true)
      expect((await setRetentionDays(request, 30)).status()).toBe(200)
      await expectWebUiState(request, 'ready')

      const pluginsResponse = await request.get('/api/plugins')
      expect(pluginsResponse.status()).toBe(200)
      const plugins = (await pluginsResponse.json()) as Array<{
        name: string
        status: string
        capabilities: string[]
      }>
      expect(plugins).toEqual(
        expect.arrayContaining([
          expect.objectContaining({
            name: pluginName,
            status: 'running',
            capabilities: expect.arrayContaining(['exemplar.notes.v1', 'mcp:tools'])
          })
        ])
      )

      const assetResponse = await request.get(`${pluginApi}/web-ui/assets/register-mesh-plugin-ui.js`)
      expect(assetResponse.status()).toBe(200)
      expect(assetResponse.headers()['content-type']).toContain('javascript')
      expect(assetResponse.headers()['cache-control']).toBe('no-cache')

      const toolResponse = await request.post(`${pluginApi}/tools/status`, { data: {} })
      expect(toolResponse.status()).toBe(200)
      expect(await toolResponse.json()).toEqual({ capability: 'exemplar.notes.v1', status: 'available' })

      const invalidConfigResponse = await setRetentionDays(request, 0)
      expect(invalidConfigResponse.status()).toBe(422)

      await page.goto('/')
      await expect(page.locator('html')).toHaveAttribute('data-theme', 'dark')
      const pluginNavLink = page.getByRole('link', { name: 'Exemplar Notes' })
      await expect(pluginNavLink).toBeVisible()
      await pluginNavLink.click()
      await expect(page).toHaveURL(new RegExp(`/plugins/${pluginName}/overview$`))
      await expect(page.getByRole('heading', { name: 'Exemplar Notes', level: 1 })).toBeVisible()
      const pluginHost = page.getByRole('region', { name: 'Exemplar Notes plugin host' })
      await expect(pluginHost.getByRole('heading', { name: 'Exemplar Notes', level: 2 })).toBeVisible()
      await expect(pluginHost).toContainText('exemplar.notes.v1 capability remains available')
      await expect(pluginHost.getByText('30 days', { exact: true })).toBeVisible()
      await expect(pluginHost.getByRole('meter', { name: 'Configured retention days' })).toHaveAttribute(
        'aria-valuenow',
        '30'
      )
      await pluginHost.getByRole('button', { name: 'Add sample note' }).click()
      await expect(pluginHost.getByRole('status')).toHaveText('Sample note 1 will be retained for 30 days.')
      await expect(page.getByText('Plugin page mounted')).toBeAttached()
      await expectBrowserApiResponse(browserApiResponses, `${pluginApi}/web-ui`)
      await expectBrowserApiResponse(browserApiResponses, `${pluginApi}/web-ui/assets/register-mesh-plugin-ui.js`)
      await page.screenshot({
        path: `${evidenceDirectory}/01-plugin-page-ready.png`,
        fullPage: true,
        animations: 'disabled'
      })

      await pluginHost.getByRole('button', { name: 'Open plugin settings' }).click()
      await expect(page).toHaveURL(/\/configuration\/plugins$/)
      await expect(page.locator('html')).toHaveAttribute('data-theme', 'dark')
      await expect(page.getByRole('heading', { name: 'Configuration', level: 1 })).toBeVisible()
      await expect(page.getByRole('tab', { name: 'Plugins' })).toHaveAttribute('aria-selected', 'true')
      const settingsBanner = page.getByRole('heading', { name: 'Plugin settings' })
      const installedPluginsHeading = page.getByRole('heading', { name: 'Installed plugins' })
      await expect(settingsBanner).toBeVisible()
      await expect(installedPluginsHeading).toBeVisible()
      const settingsBannerBox = await settingsBanner.boundingBox()
      const installedPluginsBox = await installedPluginsHeading.boundingBox()
      expect(settingsBannerBox?.y).toBeLessThan(installedPluginsBox?.y ?? Number.POSITIVE_INFINITY)
      const pluginCard = page.getByRole('article', { name: pluginName })
      await expect(pluginCard.getByText('running', { exact: true })).toBeVisible()
      await expect(pluginCard.getByText('Web UI ready', { exact: true })).toBeVisible()
      await expect(pluginCard.getByText('Assets available', { exact: true })).toBeVisible()
      await expect(pluginCard.getByRole('switch', { name: `${pluginName} web UI projection` })).toBeChecked()

      const configHost = pluginCard.getByRole('region', { name: 'Exemplar page plugin config host' })
      await expect(configHost).toContainText('Current retention: 30 days')
      await expect(configHost.getByRole('button', { name: 'Open exemplar page' })).toBeVisible()

      await page.getByRole('button', { name: /Web Ui Exemplar/ }).click()
      const retentionControl = page.getByRole('slider', { name: 'Retention days' })
      await expect(retentionControl).toHaveValue('30')
      await retentionControl.press('End')
      await expect(retentionControl).toHaveValue('365')
      const saveConfigButton = page.getByRole('button', { name: 'Save config' })
      await saveConfigButton.click()
      await expect
        .poll(async () => {
          const response = await request.get(`${pluginApi}/web-ui/config`)
          const body = (await response.json()) as { settings: { retention_days: number } }
          return body.settings.retention_days
        })
        .toBe(365)
      await expect(saveConfigButton).toHaveAccessibleName('Save config')
      await page.screenshot({
        path: `${evidenceDirectory}/02-plugin-settings-persisted.png`,
        fullPage: true,
        animations: 'disabled'
      })
      await retentionControl.scrollIntoViewIfNeeded()
      await page.screenshot({
        path: `${evidenceDirectory}/03-plugin-schema-setting.png`,
        animations: 'disabled'
      })

      await page.goto(`/plugins/${pluginName}/overview`)
      await expect(page).toHaveURL(new RegExp(`/plugins/${pluginName}/overview$`))
      const updatedPluginHost = page.getByRole('region', { name: 'Exemplar Notes plugin host' })
      await expect(updatedPluginHost.getByText('365 days', { exact: true })).toBeVisible()
      await updatedPluginHost.getByRole('button', { name: 'Add sample note' }).click()
      await expect(updatedPluginHost.getByRole('status')).toHaveText('Sample note 1 will be retained for 365 days.')

      await setWebUiEnabled(request, false)
      await expectWebUiState(request, 'disabled')
      expect((await request.get(`${pluginApi}/web-ui/assets/register-mesh-plugin-ui.js`)).status()).toBe(404)
      const toolWhileDisabled = await request.post(`${pluginApi}/tools/status`, { data: {} })
      expect(toolWhileDisabled.status()).toBe(200)
      expect(await toolWhileDisabled.json()).toEqual({ capability: 'exemplar.notes.v1', status: 'available' })

      await page.goto(`/plugins/${pluginName}/overview`)
      await expect(page.locator('html')).toHaveAttribute('data-theme', 'dark')
      await expect(page.getByRole('heading', { name: 'Plugin web UI is disabled', level: 1 })).toBeVisible()
      await page.screenshot({
        path: `${evidenceDirectory}/04-plugin-ui-disabled-capability-alive.png`,
        fullPage: true,
        animations: 'disabled'
      })

      await writeFile(
        `${evidenceDirectory}/live-validation.json`,
        `${JSON.stringify(
          {
            plugin: pluginName,
            installed: true,
            runtime_status: 'running',
            frontend_origin: new URL(page.url()).origin,
            frontend_delivery: 'production UI embedded in mesh-llm console server',
            vite_dev_client_present: consoleHtml.includes('/@vite/client'),
            api_interception: false,
            color_scheme: await page.locator('html').getAttribute('data-theme'),
            web_ui_ready: true,
            browser_page_rendered: true,
            settings_persisted: 365,
            direct_navigation_item: 'Exemplar Notes',
            page_interaction: 'Sample note 1 will be retained for 365 days.',
            invalid_setting_status: 422,
            disabled_asset_status: 404,
            non_ui_capability_while_disabled: 'available',
            asset_cache_control: 'no-cache',
            browser_api_responses: browserApiResponses,
            diagnostics
          },
          null,
          2
        )}\n`
      )

      expect(diagnostics.consoleErrors, 'unexpected browser console errors').toHaveLength(0)
      expect(diagnostics.pageErrors, 'uncaught browser exceptions').toHaveLength(0)
    } finally {
      await setWebUiEnabled(request, true)
      expect((await setRetentionDays(request, 30)).status()).toBe(200)
    }
  })
})
