// The page mounts inside the console's own DOM. Its injected stylesheet must
// not restyle the console: the console's `<nav class="hidden md:flex">` keeps
// its display while the page is mounted.
import { act } from '@testing-library/react'
import { afterEach, expect, it, vi } from 'vitest'
import { EVIDENCE_PAGE_ID, registerMeshPluginUi } from '@/register-mesh-plugin-ui'
import { setPluginHost } from '@/plugin-host/host'
import { createStandaloneHost } from '@/plugin-host/standalone-host'
import { portalContainer } from '@/lib/scope'

vi.mock('@/features/capsules/pages/EvidencePage', async (importOriginal) => {
  const actual = await importOriginal<typeof import('@/features/capsules/pages/EvidencePage')>()
  return { ...actual, EvidencePage: () => <p className="hidden" data-testid="page-hidden">page</p> }
})

afterEach(() => {
  setPluginHost(createStandaloneHost())
  document.body.replaceChildren()
})

/** jsdom does not apply rules inside `@layer`. Copy the injected sheet's style
 *  rules, at any depth, into one flat sheet so that computed styles show what
 *  they would do in a browser. */
function flattenInjectedSheet(): void {
  const injected = document.getElementById('capsules-evidence-styles') as HTMLStyleElement
  const rules: string[] = []
  const walk = (list: CSSRuleList) => {
    for (const rule of Array.from(list)) {
      if (rule instanceof CSSStyleRule) rules.push(rule.cssText)
      else if ('cssRules' in rule && rule.constructor.name === 'CSSLayerBlockRule') walk((rule as CSSGroupingRule).cssRules)
    }
  }
  walk(injected.sheet!.cssRules)
  const flat = document.createElement('style')
  flat.textContent = rules.join('\n')
  document.head.appendChild(flat)
}

/** A host-like DOM: the console's header nav and some console elements using
 *  the same utility names the page's sheet carries. */
function consoleDom(): { nav: HTMLElement; others: HTMLElement[]; mount: HTMLElement } {
  document.body.innerHTML = `
    <header class="flex items-center">
      <nav class="hidden md:flex gap-1" data-testid="console-nav"><a class="px-2">Network</a></nav>
      <span class="sr-only">label</span>
    </header>
    <main><section aria-label="Evidence plugin host"></section></main>`
  return {
    nav: document.querySelector('[data-testid="console-nav"]')!,
    others: Array.from(document.querySelectorAll<HTMLElement>('header, header *, main, section')),
    mount: document.querySelector('section')!
  }
}

it("the console's nav keeps its computed display, and no rule of the page's matches a console element", async () => {
  const { nav, others, mount } = consoleDom()
  const before = getComputedStyle(nav).display
  const host = createStandaloneHost()
  const registration = await registerMeshPluginUi(host)
  let handle: { unmount: () => void } | undefined
  await act(async () => {
    handle = await registration.pages[EVIDENCE_PAGE_ID]!({ element: mount, host, page: host.webUi.pages![0]! })
  })
  flattenInjectedSheet()

  expect(getComputedStyle(nav).display).toBe(before)
  expect(getComputedStyle(nav).display).not.toBe('none')
  // The same utility does apply inside the page, so the sheet is live.
  expect(getComputedStyle(mount.querySelector('[data-testid="page-hidden"]')!).display).toBe('none')
  // And inside the page's portal container (popovers, menus, dialogs).
  const portaled = document.createElement('div')
  portaled.className = 'hidden'
  portalContainer().appendChild(portaled)
  expect(getComputedStyle(portaled).display).toBe('none')

  const injected = (document.getElementById('capsules-evidence-styles') as HTMLStyleElement).sheet!
  const selectors: string[] = []
  const walk = (list: CSSRuleList) => {
    for (const rule of Array.from(list)) {
      if (rule instanceof CSSStyleRule) selectors.push(rule.selectorText)
      else if ('cssRules' in rule) walk((rule as CSSGroupingRule).cssRules)
    }
  }
  walk(injected.cssRules)
  expect(selectors.length).toBeGreaterThan(100)
  const matching = (element: Element) =>
    selectors.filter((selector) => {
      try {
        return element.matches(selector)
      } catch {
        return false
      }
    })
  // The matching itself works: the page's own `.hidden` element is matched.
  expect(matching(mount.querySelector('[data-testid="page-hidden"]')!)).not.toEqual([])
  for (const element of others) expect([element.tagName, matching(element)]).toEqual([element.tagName, []])

  await act(async () => handle!.unmount())
})
