import { createRootRoute, createRoute, createRouter, lazyRouteComponent } from '@tanstack/react-router'
import { AppErrorBoundary, NotFoundRoute } from '@/app/error-boundaries/AppErrorBoundary'
import { FeatureErrorBoundary } from '@/app/error-boundaries/FeatureErrorBoundary'
import { RootLayout } from '@/app/layout/RootLayout'
import { parseDeveloperPlaygroundSearch } from '@/features/developer/playground/developer-playground-tabs'
import { parseLogsLedgerSearch } from '@/features/logs/lib/log-search'
import { parseLogRequestDetailsSearch } from '@/features/logs/lib/log-request-details'
import { LogsFeatureGate } from '@/features/logs/pages/LogsFeatureGate'
import { env } from '@/lib/env'

const enableMeshVizPerfRoute = env.isDevelopment || import.meta.env.VITE_ENABLE_PERF_ROUTE === 'true'

const rootRoute = createRootRoute({
  component: RootLayout,
  errorComponent: AppErrorBoundary,
  notFoundComponent: NotFoundRoute
})
const indexRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: '/',
  head: () => ({ meta: [{ title: 'MeshLLM - Dashboard' }] }),
  component: lazyRouteComponent(() => import('@/features/network/pages/DashboardPage'), 'DashboardPageSurface'),
  errorComponent: FeatureErrorBoundary
})
const reservesRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: '/reserves',
  head: () => ({ meta: [{ title: 'MeshLLM - Reserves' }] }),
  component: lazyRouteComponent(() => import('@/features/reserves/pages/ReservesPage'), 'ReservesPageContent'),
  errorComponent: FeatureErrorBoundary
})
const LogsLedgerPage = lazyRouteComponent(() => import('@/features/logs/pages/LogsLedgerPage'), 'LogsLedgerPage')
const LogRequestDetailsPage = lazyRouteComponent(
  () => import('@/features/logs/pages/LogRequestDetailsPage'),
  'LogRequestDetailsPage'
)
const logsRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: '/logs',
  head: () => ({ meta: [{ title: 'MeshLLM - Logs' }] }),
  validateSearch: parseLogsLedgerSearch,
  component: () => (
    <LogsFeatureGate>
      <LogsLedgerPage />
    </LogsFeatureGate>
  ),
  errorComponent: FeatureErrorBoundary
})
const logRequestDetailsRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: '/logs/$requestId',
  head: () => ({ meta: [{ title: 'MeshLLM - Request details' }] }),
  validateSearch: parseLogRequestDetailsSearch,
  component: () => (
    <LogsFeatureGate>
      <LogRequestDetailsPage />
    </LogsFeatureGate>
  ),
  errorComponent: FeatureErrorBoundary
})
const chatRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: '/chat',
  head: () => ({ meta: [{ title: 'MeshLLM - Chat' }] }),
  component: lazyRouteComponent(() => import('@/features/chat/pages/ChatPage'), 'ChatPageContent'),
  errorComponent: FeatureErrorBoundary
})
const configurationRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: '/configuration',
  head: () => ({ meta: [{ title: 'MeshLLM - Configuration' }] }),
  component: lazyRouteComponent(
    () => import('@/features/configuration/pages/ConfigurationRoutePage'),
    'ConfigurationRoutePage'
  ),
  errorComponent: FeatureErrorBoundary
})
const configurationTabRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: '/configuration/$configurationTab',
  head: () => ({ meta: [{ title: 'MeshLLM - Configuration' }] }),
  component: lazyRouteComponent(
    () => import('@/features/configuration/pages/ConfigurationRoutePage'),
    'ConfigurationRoutePage'
  ),
  errorComponent: FeatureErrorBoundary
})
const pluginWebUiRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: '/plugins/$pluginName/$pageId',
  head: () => ({ meta: [{ title: 'MeshLLM - Plugin' }] }),
  component: lazyRouteComponent(() => import('@/features/plugins/web-ui/PluginWebUiRoutePage'), 'PluginWebUiRoutePage'),
  errorComponent: FeatureErrorBoundary
})
const developerPlaygroundRoute = env.isDevelopment
  ? createRoute({
      getParentRoute: () => rootRoute,
      path: '/__playground',
      head: () => ({ meta: [{ title: 'MeshLLM - Developer Playground' }] }),
      validateSearch: parseDeveloperPlaygroundSearch,
      component: lazyRouteComponent(
        () => import('@/features/developer/pages/DeveloperPlaygroundPage'),
        'DeveloperPlaygroundPage'
      )
    })
  : null
const meshVizPerfRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: '/__meshviz-perf',
  component: lazyRouteComponent(() => import('@/features/network/pages/MeshVizPerfPage'), 'MeshVizPerfPage')
})
export const routeTree = rootRoute.addChildren([
  indexRoute,
  reservesRoute,
  logsRoute,
  logRequestDetailsRoute,
  chatRoute,
  configurationRoute,
  configurationTabRoute,
  pluginWebUiRoute,
  ...(developerPlaygroundRoute ? [developerPlaygroundRoute] : []),
  ...(enableMeshVizPerfRoute ? [meshVizPerfRoute] : [])
])
export const router = createRouter({ routeTree, basepath: env.routerBasePath })
declare module '@tanstack/react-router' {
  interface Register {
    router: typeof router
  }
}
