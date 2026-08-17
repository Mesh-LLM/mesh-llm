import { defineConfig, devices } from '@playwright/test'

const host = process.env.PLAYWRIGHT_HOST ?? '127.0.0.1'
const port = Number(process.env.PLAYWRIGHT_PORT ?? 51973)
// The regular suite drives Vite. The logging certification harness points the
// same Playwright tests at the console embedded in a real mesh-llm process.
const externalBaseURL = process.env.MESH_LOGS_E2E_BASE_URL
const baseURL = externalBaseURL ?? `http://${host}:${port}`
const jsonReport = process.env.PLAYWRIGHT_JSON_REPORT

export default defineConfig({
  testDir: './e2e',
  fullyParallel: true,
  forbidOnly: Boolean(process.env.CI),
  retries: process.env.CI ? 2 : 0,
  workers: process.env.CI ? 1 : undefined,
  reporter: jsonReport ? [['list'], ['json', { outputFile: jsonReport }]] : 'list',
  outputDir: process.env.PLAYWRIGHT_OUTPUT_DIR ?? 'test-results',
  use: { baseURL, screenshot: 'only-on-failure', trace: 'on-first-retry' },
  projects: [{ name: 'chromium', use: { ...devices['Desktop Chrome'] } }],
  webServer: externalBaseURL
    ? undefined
    : {
        command: `pnpm exec vite --host ${host} --port ${port} --strictPort`,
        url: baseURL,
        reuseExistingServer: !process.env.CI,
        timeout: 30_000,
        stdout: 'pipe',
        stderr: 'pipe'
      }
})
