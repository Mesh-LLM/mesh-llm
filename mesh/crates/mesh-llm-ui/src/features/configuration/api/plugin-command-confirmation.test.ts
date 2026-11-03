// @vitest-environment jsdom
import { afterEach, expect, it, vi } from 'vitest'
import { confirmPluginCommands } from './plugin-command-confirmation'

afterEach(() => {
  document.body.replaceChildren()
  vi.restoreAllMocks()
})

it('shows the complete command as text and does not approve on cancellation', async () => {
  Object.defineProperty(HTMLDialogElement.prototype, 'showModal', { configurable: true, value: vi.fn() })
  Object.defineProperty(HTMLDialogElement.prototype, 'close', { configurable: true, value: vi.fn() })
  const command = 'scanner: /bin/scan ["<script>alert(1)</script>"]'
  const result = confirmPluginCommands([command])
  expect(document.querySelector('pre')?.textContent).toBe(command)
  expect(document.querySelector('script')).toBeNull()
  ;(document.querySelector('button') as HTMLButtonElement).click()
  expect(await result).toBe(false)
})
