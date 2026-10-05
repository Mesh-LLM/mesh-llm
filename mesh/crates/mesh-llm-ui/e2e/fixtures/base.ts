import { devices, expect, test as base } from '@playwright/test'

export { devices, expect }
export type { APIRequestContext, Page, TestInfo } from '@playwright/test'

export const test = base.extend({
  page: async ({ page }, applyFixture) => {
    if (!process.env.MESH_PLUGIN_E2E) {
      await page.route('**/api/plugins', (route) => route.fulfill({ json: [] }))
    }
    await applyFixture(page)
  }
})
