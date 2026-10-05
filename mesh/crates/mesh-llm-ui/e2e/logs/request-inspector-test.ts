import { expect, test as base } from '../fixtures/base'

type BrowserIssueFixture = {
  readonly browserIssueGuard: void
}

export const test = base.extend<BrowserIssueFixture>({
  browserIssueGuard: [
    async ({ page }, use) => {
      const issues: string[] = []
      await page.route('https://fonts.googleapis.com/**', (route) =>
        route.fulfill({ body: '', contentType: 'text/css', status: 200 })
      )
      page.on('pageerror', (error) => issues.push(`pageerror: ${error.message}`))
      page.on('console', (message) => {
        if (message.type() === 'error') issues.push(`console: ${message.text()}`)
      })
      page.on('response', (response) => {
        if (response.status() === 404) issues.push(`response: 404 ${response.url()}`)
      })

      await use()

      expect(issues).toEqual([])
    },
    { auto: true }
  ]
})

export { expect }
