import type { Page } from '@playwright/test'
import {
  requestDeleteReceipt,
  REQUEST_INSPECTOR_ARTIFACT_DETAILS,
  REQUEST_INSPECTOR_ARTIFACT_IDS,
  REQUEST_INSPECTOR_IDS,
  REQUEST_INSPECTOR_SCENARIOS,
  REQUEST_INSPECTOR_SHELL_STATUS,
  REQUEST_INSPECTOR_STREAM_HOSTILE_TEXT
} from './request-inspector-fixtures'

const DATA_MODE_STORAGE_KEY = 'mesh-llm-ui-preview:data-mode:v2'

export { REQUEST_INSPECTOR_ARTIFACT_IDS, REQUEST_INSPECTOR_IDS, REQUEST_INSPECTOR_STREAM_HOSTILE_TEXT }

type RequestCapability = 'supported' | 'unsupported' | 'loading'

type RequestInspectorRouteOptions = {
  readonly capability?: RequestCapability
}

export type RequestInspectorBackendEvidence = {
  readonly auditStreamCalls: number
  readonly requestListCalls: number
  readonly logStreamCalls: number
  readonly summaryCalls: readonly string[]
  readonly eventCalls: readonly string[]
  readonly artifactListCalls: readonly string[]
  readonly proxyCalls: readonly string[]
  readonly artifactDetailCalls: readonly string[]
  readonly deleteRequestBodies: readonly string[]
  readonly releaseCapability: () => void
}

function logsPage(items: readonly object[]) {
  return { items, nextCursor: null }
}

export async function installRequestInspectorRoutes(
  page: Page,
  options: RequestInspectorRouteOptions = {}
): Promise<RequestInspectorBackendEvidence> {
  await page.addInitScript((storageKey) => window.localStorage.setItem(storageKey, 'live'), DATA_MODE_STORAGE_KEY)
  const capability = options.capability ?? 'supported'
  const summaryCalls: string[] = []
  const eventCalls: string[] = []
  const artifactListCalls: string[] = []
  const proxyCalls: string[] = []
  const artifactDetailCalls: string[] = []
  const deleteRequestBodies: string[] = []
  let requestListCalls = 0
  let auditStreamCalls = 0
  let logStreamCalls = 0
  let releaseCapability = () => undefined
  const capabilityGate = new Promise<void>((resolve) => {
    releaseCapability = resolve
  })

  await page.route('**/api/status', (route) => route.fulfill({ json: REQUEST_INSPECTOR_SHELL_STATUS }))
  await page.route('**/api/events', (route) =>
    route.fulfill({ contentType: 'text/event-stream', body: 'retry: 60000\n\n' })
  )
  await page.context().route('**/api/logs/**', async (route) => {
    const url = new URL(route.request().url())
    const method = route.request().method()

    if (url.pathname === '/api/logs/requests' && method === 'GET') {
      requestListCalls += 1
      if (capability === 'loading') await capabilityGate
      if (capability === 'unsupported') {
        await route.fulfill({ status: 404, json: { error: { code: 'unsupported' } } })
        return
      }
      await route.fulfill({
        json: logsPage(Object.values(REQUEST_INSPECTOR_SCENARIOS).map((scenario) => scenario.summary))
      })
      return
    }
    if (url.pathname === '/api/logs/audit' && method === 'GET') {
      await route.fulfill({ json: logsPage([]) })
      return
    }
    if (url.pathname === '/api/logs/events' && method === 'GET') {
      if (url.searchParams.get('audit') === '1') auditStreamCalls += 1
      else logStreamCalls += 1
      await route.fulfill({ contentType: 'text/event-stream', body: 'retry: 60000\n\n' })
      return
    }
    if (url.pathname === '/api/logs/proxy' && method === 'GET') {
      const requestId = url.searchParams.get('request_id') ?? ''
      proxyCalls.push(requestId)
      await route.fulfill({ json: logsPage(REQUEST_INSPECTOR_SCENARIOS[requestId]?.attempts ?? []) })
      return
    }

    const artifactMatch = /^\/api\/logs\/artifacts\/([^/]+)$/.exec(url.pathname)
    if (artifactMatch?.[1] && method === 'GET') {
      const artifactId = artifactMatch[1]
      artifactDetailCalls.push(artifactId)
      const detail = REQUEST_INSPECTOR_ARTIFACT_DETAILS[artifactId]
      await route.fulfill(detail ? { json: detail } : { status: 404, json: { error: { code: 'not_found' } } })
      return
    }

    const deleteMatch = /^\/api\/logs\/requests\/([^/]+)\/delete$/.exec(url.pathname)
    if (deleteMatch?.[1] && method === 'POST') {
      deleteRequestBodies.push(route.request().postData() ?? '')
      await route.fulfill({ json: requestDeleteReceipt(deleteMatch[1]) })
      return
    }

    const requestMatch = /^\/api\/logs\/requests\/([^/]+)(?:\/(events|artifacts))?$/.exec(url.pathname)
    const requestId = requestMatch?.[1]
    const scenario = requestId ? REQUEST_INSPECTOR_SCENARIOS[requestId] : undefined
    if (requestId && scenario && method === 'GET') {
      const resource = requestMatch?.[2]
      if (resource === 'events') eventCalls.push(requestId)
      else if (resource === 'artifacts') artifactListCalls.push(requestId)
      else summaryCalls.push(requestId)
      await route.fulfill({
        json:
          resource === 'events'
            ? logsPage(scenario.events)
            : resource === 'artifacts'
              ? logsPage(scenario.artifacts)
              : scenario.summary
      })
      return
    }
    await route.fulfill({ status: 404, json: { error: { code: 'unsupported' } } })
  })

  return {
    get auditStreamCalls() {
      return auditStreamCalls
    },
    get requestListCalls() {
      return requestListCalls
    },
    get logStreamCalls() {
      return logStreamCalls
    },
    summaryCalls,
    eventCalls,
    artifactListCalls,
    proxyCalls,
    artifactDetailCalls,
    deleteRequestBodies,
    releaseCapability
  }
}
