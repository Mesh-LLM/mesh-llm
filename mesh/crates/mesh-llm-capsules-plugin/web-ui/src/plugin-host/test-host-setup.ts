// Every unit test runs with a mounted standalone host, the way every page runs
// with the console's host: the data clients resolve plugin-relative paths to
// `/api/plugins/capsules/...` and call the (stubbable) global `fetch`.
import { setPluginHost } from '@/plugin-host/host'
import { createStandaloneHost } from '@/plugin-host/standalone-host'

setPluginHost(createStandaloneHost())
