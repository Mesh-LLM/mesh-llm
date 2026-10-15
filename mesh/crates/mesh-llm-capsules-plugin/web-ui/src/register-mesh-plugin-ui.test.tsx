import { act } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'
import {
  CHAT_CONTRIBUTION_ID,
  EVIDENCE_PAGE_ID,
  LOGS_CONTRIBUTION_ID,
  registerMeshPluginUi
} from '@/register-mesh-plugin-ui'
import { pluginHost, setPluginHost } from '@/plugin-host/host'
import { createStandaloneHost } from '@/plugin-host/standalone-host'

vi.mock('@/features/capsules/pages/EvidencePage', async (importOriginal) => {
  const actual = await importOriginal<typeof import('@/features/capsules/pages/EvidencePage')>()
  return {
    ...actual,
    EvidencePage: ({ focusExchangeKey }: { focusExchangeKey?: string }) => (
      <p data-testid="evidence-page">focus:{focusExchangeKey ?? 'none'}</p>
    )
  }
})

const STYLE_ID = 'capsules-evidence-styles'

afterEach(() => {
  // Every other test runs with the setup file's standalone host mounted.
  setPluginHost(createStandaloneHost())
  window.history.replaceState(null, '', '/')
})

describe('registerMeshPluginUi', () => {
  it('registers exactly one page, `evidence`, and no config sections', async () => {
    const registration = await registerMeshPluginUi(createStandaloneHost())
    expect(Object.keys(registration.pages)).toEqual([EVIDENCE_PAGE_ID])
    expect(Object.keys(registration.contributions ?? {})).toEqual([CHAT_CONTRIBUTION_ID, LOGS_CONTRIBUTION_ID])
    expect(registration.configSections).toBeUndefined()
  })

  it('mounts into the given element, holds the host, and injects one stylesheet', async () => {
    const host = createStandaloneHost()
    const element = document.createElement('section')
    document.body.appendChild(element)
    const registration = await registerMeshPluginUi(host)
    let handle: Awaited<ReturnType<(typeof registration.pages)[string]>> | undefined
    await act(async () => {
      handle = await registration.pages[EVIDENCE_PAGE_ID]!({ element, host, page: host.webUi.pages![0]! })
    })

    expect(element.querySelector('[data-testid="evidence-page"]')).not.toBeNull()
    expect(pluginHost()).toBe(host)
    expect(document.querySelectorAll(`#${STYLE_ID}`)).toHaveLength(1)
    expect(document.getElementById(STYLE_ID)?.textContent).toContain('bg-background')

    await act(async () => handle!.unmount())
    expect(element.childElementCount).toBe(0)
    expect(document.getElementById(STYLE_ID)).toBeNull()
    expect(() => pluginHost()).toThrow(/no mesh-llm host is mounted/)
    element.remove()
  })

  it('a remount replaces the stylesheet rather than stacking a second one', async () => {
    const host = createStandaloneHost()
    const registration = await registerMeshPluginUi(host)
    const handles: Array<{ unmount: () => void }> = []
    await act(async () => {
      for (const element of [document.createElement('div'), document.createElement('div')]) {
        handles.push(await registration.pages[EVIDENCE_PAGE_ID]!({ element, host, page: host.webUi.pages![0]! }))
      }
    })
    expect(document.querySelectorAll(`#${STYLE_ID}`)).toHaveLength(1)
    await act(async () => handles.forEach((handle) => handle.unmount()))
  })

  it('reads the per-row deep link from the page URL (`?focusExchangeKey=`)', async () => {
    window.history.replaceState(null, '', '/plugins/capsules/evidence?focusExchangeKey=exch-1')
    const host = createStandaloneHost()
    const element = document.createElement('div')
    const registration = await registerMeshPluginUi(host)
    let handle: { unmount: () => void } | undefined
    await act(async () => {
      handle = await registration.pages[EVIDENCE_PAGE_ID]!({ element, host, page: host.webUi.pages![0]! })
    })
    expect(element.textContent).toBe('focus:exch-1')
    await act(async () => handle!.unmount())
  })
})
