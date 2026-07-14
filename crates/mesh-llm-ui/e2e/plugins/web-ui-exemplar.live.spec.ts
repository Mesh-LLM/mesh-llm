import { mkdir, writeFile } from 'node:fs/promises'
import { resolve } from 'node:path'
import { expect, test, type APIRequestContext, type Page } from '@playwright/test'

const pluginName = 'web-ui-exemplar'
const pluginApi = `/api/plugins/${pluginName}`
const evidenceDirectory = resolve(
  process.env.MESH_PLUGIN_EVIDENCE_DIR ?? '../../../target/plugin-web-ui-evidence/playwright'
)

type BrowserDiagnostics = {
  consoleErrors: string[]
  pageErrors: string[]
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

test.describe('installed plugin web UI exemplar @plugin', () => {
  test.skip(!process.env.MESH_PLUGIN_E2E, 'requires the installed live exemplar started by its documented recipe')
  test.describe.configure({ mode: 'serial' })

  test('is installed, running, rendered, configurable, and independently disableable', async ({ page, request }) => {
    const diagnostics = collectBrowserDiagnostics(page)
    await mkdir(evidenceDirectory, { recursive: true })

    try {
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

      await page.goto(`/plugins/${pluginName}/overview`)
      await expect(page.getByRole('heading', { name: 'Exemplar Overview', level: 1 })).toBeVisible()
      const pluginHost = page.getByRole('region', { name: 'Exemplar Overview plugin host' })
      await expect(pluginHost.getByRole('heading', { name: 'Exemplar Overview', level: 2 })).toBeVisible()
      await expect(pluginHost).toContainText('exemplar.notes.v1 remains available')
      await expect(page.getByText('Plugin page mounted')).toBeAttached()
      await page.screenshot({
        path: `${evidenceDirectory}/01-plugin-page-ready.png`,
        fullPage: true,
        animations: 'disabled'
      })

      await page.goto('/configuration/plugins')
      await expect(page.getByRole('heading', { name: 'Configuration', level: 1 })).toBeVisible()
      await expect(page.getByRole('tab', { name: 'Plugins' })).toHaveAttribute('aria-selected', 'true')
      const pluginCard = page.getByRole('article', { name: pluginName })
      await expect(pluginCard.getByText('running', { exact: true })).toBeVisible()
      await expect(pluginCard.getByText('Web UI ready', { exact: true })).toBeVisible()
      await expect(pluginCard.getByText('Assets available', { exact: true })).toBeVisible()
      await expect(pluginCard.getByRole('switch', { name: `${pluginName} web UI projection` })).toBeChecked()

      const configHost = pluginCard.getByRole('region', { name: 'Exemplar Retention plugin config host' })
      const retentionInput = configHost.getByRole('spinbutton')
      await expect(retentionInput).toHaveValue('30')
      await retentionInput.fill('46')
      await configHost.getByRole('button', { name: 'Save retention' }).click()
      await expect(pluginCard.getByRole('status')).toContainText(/Retention saved|Plugin settings saved/)
      await expect
        .poll(async () => {
          const response = await request.get(`${pluginApi}/web-ui/config`)
          const body = (await response.json()) as { settings: { retention_days: number } }
          return body.settings.retention_days
        })
        .toBe(46)
      await page.reload()
      await expect(
        page
          .getByRole('article', { name: pluginName })
          .getByRole('region', { name: 'Exemplar Retention plugin config host' })
          .getByRole('spinbutton')
      ).toHaveValue('46')
      await page.screenshot({
        path: `${evidenceDirectory}/02-plugin-settings-persisted.png`,
        fullPage: true,
        animations: 'disabled'
      })

      await setWebUiEnabled(request, false)
      await expectWebUiState(request, 'disabled')
      expect((await request.get(`${pluginApi}/web-ui/assets/register-mesh-plugin-ui.js`)).status()).toBe(404)
      const toolWhileDisabled = await request.post(`${pluginApi}/tools/status`, { data: {} })
      expect(toolWhileDisabled.status()).toBe(200)
      expect(await toolWhileDisabled.json()).toEqual({ capability: 'exemplar.notes.v1', status: 'available' })

      await page.goto(`/plugins/${pluginName}/overview`)
      await expect(page.getByRole('heading', { name: 'Plugin web UI is disabled', level: 1 })).toBeVisible()
      await page.screenshot({
        path: `${evidenceDirectory}/03-plugin-ui-disabled-capability-alive.png`,
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
            web_ui_ready: true,
            browser_page_rendered: true,
            settings_persisted: 46,
            invalid_setting_status: 422,
            disabled_asset_status: 404,
            non_ui_capability_while_disabled: 'available',
            asset_cache_control: 'no-cache',
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
