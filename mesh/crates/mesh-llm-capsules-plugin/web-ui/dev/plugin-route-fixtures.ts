// Fixture mode for the plugin page (dev server only; never in the bundle).
//
// `VITE_EVIDENCE_FIXTURES=<dir>` answers the page's plugin routes
// (`/api/plugins/capsules/<route>`) and the two console routes it reads
// (`/api/status`, `/api/models`) from a fixture set captured by the fork
// console's `scripts/capture-evidence-fixtures.mjs` -- e.g.
// `<captures>/freeze-candidate`. Those sets are keyed by
// the fork's host routes (`/api/capsules/panes/*`, `/api/capsules/ledger/*`),
// so each plugin route is translated to the host route that captured the same
// data, and wrapped the way the plugin serves it (JSON only).
//
// A route with no captured response is answered 404 with a JSON body naming
// the miss, never synthesised: a fixture that invents data would hide exactly
// the defects the page exists to show.
import fs from 'node:fs'
import path from 'node:path'
import type { IncomingMessage, ServerResponse } from 'node:http'
import type { PluginOption } from 'vite'

export const PLUGIN_API_PREFIX = '/api/plugins/capsules/'

type FixtureEntry = { file: string; status: number; contentType: string }
type FixtureManifest = { entries: Record<string, FixtureEntry> }

export type Captured = { status: number; contentType: string; body: Buffer }
/** Reads one captured response by its manifest key (`GET /api/...`), or null. */
export type CaptureReader = (key: string) => Captured | null

export type RouteAnswer = { status: number; contentType: string; body: string | Buffer }

function json(status: number, value: unknown): RouteAnswer {
  return { status, contentType: 'application/json', body: JSON.stringify(value) }
}

function miss(request: string, capturedAs: string): RouteAnswer {
  return json(404, { error: 'evidence fixture mode: not captured', request, captured_as: capturedAs })
}

function parseJsonl(text: string): unknown[] {
  const records: unknown[] = []
  for (const line of text.split('\n')) {
    const trimmed = line.trim()
    if (!trimmed) continue
    try {
      records.push(JSON.parse(trimmed))
    } catch {
      // The plugin's own ledger route skips a malformed line the same way.
    }
  }
  return records
}

/** Answers one page request (`method`, path + query) from the captured set. */
export function answerFromFixtures(method: string, url: string, read: CaptureReader): RouteAnswer | null {
  const parsed = new URL(url, 'http://fixture.invalid')
  const request = `${method} ${parsed.pathname}${parsed.search}`

  if (parsed.pathname === '/api/status' || parsed.pathname === '/api/models') {
    const key = `GET ${parsed.pathname}`
    const hit = method === 'GET' ? read(key) : null
    return hit ? { status: hit.status, contentType: hit.contentType, body: hit.body } : miss(request, key)
  }
  if (!parsed.pathname.startsWith(PLUGIN_API_PREFIX)) return null
  // The page's data routes are the plugin's HTTP bindings, which the console
  // serves under `http/`; a tool call is not.
  const pluginPath = parsed.pathname.slice(PLUGIN_API_PREFIX.length)
  const route = pluginPath.startsWith('http/') ? pluginPath.slice('http/'.length) : pluginPath

  // Recorded captures predate the door route: a captured node had its door.
  if (method === 'GET' && route === 'door') return json(200, { state: 'ready', url: 'fixture' })

  if (method === 'GET' && route.startsWith('panes/')) {
    const key = `GET /api/capsules/${route}${parsed.search}`
    const hit = read(key)
    return hit ? { status: hit.status, contentType: hit.contentType, body: hit.body } : miss(request, key)
  }

  if (method === 'GET' && route === 'ledger') {
    const ledgerKey = 'GET /api/capsules/ledger/capsules.jsonl'
    const ledger = read(ledgerKey)
    if (!ledger) return miss(request, ledgerKey)
    if (ledger.status !== 200) return json(ledger.status, { error: 'captured ledger was not 200' })
    const key = read('GET /api/capsules/ledger/node-key.pub.pem')
    return json(200, {
      records: parseJsonl(ledger.body.toString('utf8')),
      node_pub_key_pem: key && key.status === 200 ? key.body.toString('utf8') : null
    })
  }

  const capsuleId = parsed.searchParams.get('capsule_id')
  if (method === 'GET' && route === 'ledger/signed-statement' && capsuleId) {
    const key = `GET /api/capsules/ledger/signed-statements/${encodeURIComponent(capsuleId)}.cose`
    const hit = read(key)
    if (!hit) return miss(request, key)
    return json(200, { signed_statement_b64: hit.status === 200 ? hit.body.toString('base64') : null })
  }
  if (method === 'GET' && route === 'ledger/disclosure' && capsuleId) {
    const key = `GET /api/capsules/ledger/disclosures/${encodeURIComponent(capsuleId)}.json`
    const hit = read(key)
    if (!hit) return miss(request, key)
    return json(200, { disclosure: hit.status === 200 ? JSON.parse(hit.body.toString('utf8')) : null })
  }

  // Any other plugin route (e.g. `tools/mesh_ledger_fetch`) was never
  // captured by the fork console under a plugin path.
  return miss(request, '(no captured equivalent)')
}

export function directoryReader(dir: string): CaptureReader {
  return (key) => {
    const manifestFile = path.join(dir, 'manifest.json')
    if (!fs.existsSync(manifestFile)) return null
    // Re-read per request so a re-capture shows up without restarting Vite.
    const manifest = JSON.parse(fs.readFileSync(manifestFile, 'utf8')) as FixtureManifest
    const entry = manifest.entries[key]
    if (!entry) return null
    return { status: entry.status, contentType: entry.contentType, body: fs.readFileSync(path.join(dir, entry.file)) }
  }
}

export function pluginRouteFixtures(dir: string | undefined): PluginOption {
  return {
    name: 'capsules:plugin-route-fixtures',
    configureServer(server) {
      if (!dir) return
      const resolved = path.resolve(dir)
      const read = directoryReader(resolved)
      const logged = new Set<string>()
      server.config.logger.info(`[evidence-fixtures] answering plugin routes from ${resolved}`)
      server.middlewares.use((req: IncomingMessage, res: ServerResponse, next: () => void) => {
        if (!req.url?.startsWith('/api/')) return next()
        const answer = answerFromFixtures(req.method ?? 'GET', req.url, read)
        if (!answer) return next()
        if (answer.status === 404 && !logged.has(req.url)) {
          logged.add(req.url)
          server.config.logger.warn(`[evidence-fixtures] no fixture for ${req.method} ${req.url}`)
        }
        res.statusCode = answer.status
        res.setHeader('content-type', answer.contentType)
        res.end(answer.body)
      })
    }
  }
}
