'use strict'

const path = require('node:path')
const nativeRuntime = require('./native-runtime')
const { Inference, OpenAIRequestError } = require('./inference')
const nativeModuleCache = new Map()

function loadNativeAddon() {
  const explicit = process.env.MESHLLM_NODE_NATIVE_PATH
  if (explicit) return loadNativeFile(explicit)

  const platformArch = `${process.platform}-${process.arch}`
  const candidates = [
    path.join(__dirname, 'native', platformArch, 'mesh_llm_nodejs.node'),
    path.join(__dirname, 'native', 'mesh_llm_nodejs.node'),
    path.join(__dirname, '..', '..', 'target', 'release', nativeAddonName()),
    path.join(__dirname, '..', '..', 'target', 'debug', nativeAddonName())
  ]

  const errors = []
  for (const candidate of candidates) {
    try {
      return loadNativeFile(candidate)
    } catch (error) {
      if (error && error.code !== 'MODULE_NOT_FOUND') errors.push(`${candidate}: ${error.message}`)
    }
  }

  throw new Error(
    `MeshLLM Node native addon was not found for ${platformArch}. ` +
    `Run npm run build:native, install a package with prebuilt native assets, ` +
    `or set MESHLLM_NODE_NATIVE_PATH. ${errors.join('; ')}`
  )
}

function loadNativeFile(file) {
  const resolved = path.resolve(file)
  const cached = nativeModuleCache.get(resolved)
  if (cached) return cached
  const mod = { exports: {} }
  // process.dlopen lets development builds load the platform library directly
  // from target/{debug,release}; cache the exports so repeated SDK imports do
  // not initialize the native addon more than once for the same resolved path.
  process.dlopen(mod, resolved)
  nativeModuleCache.set(resolved, mod.exports)
  return mod.exports
}

function nativeAddonName() {
  if (process.platform === 'win32') return 'mesh_llm_nodejs.dll'
  if (process.platform === 'darwin') return 'libmesh_llm_nodejs.dylib'
  return 'libmesh_llm_nodejs.so'
}

const native = loadNativeAddon()
nativeRuntime.configureNativeRuntimeBinding(native)

class Node {
  constructor(handle) {
    this._handle = handle
    this.inference = new Inference(handle)
  }

  static create(options = {}) {
    const handle = native.Node.create(
      options.mode || 'client',
      options.joinTokens || [],
      options.models || [],
      options.autoJoin === true,
      options.ownerKeyPath || null,
      options.apiPort || 9337,
      options.consolePort || 3131
    )
    return new Node(handle)
  }

  start() {
    return this._handle.start()
  }

  stop() {
    return this._handle.stop()
  }

  async status() {
    return parse(await this._handle.statusJson())
  }

  joinToken(token) {
    return this._handle.joinToken(token)
  }
}

function parse(json) {
  return JSON.parse(json)
}

module.exports = {
  Inference,
  Node,
  OpenAIRequestError,
  currentMeshVersion: nativeRuntime.currentMeshVersion,
  currentSkippyAbiVersion: nativeRuntime.currentSkippyAbiVersion,
  installNativeRuntime: nativeRuntime.installNativeRuntime,
  installedNativeRuntimes: nativeRuntime.installedNativeRuntimes,
  removeNativeRuntime: nativeRuntime.removeNativeRuntime,
  pruneNativeRuntimes: nativeRuntime.pruneNativeRuntimes,
  resolveNativeRuntime: nativeRuntime.resolveNativeRuntime,
  validateNativeRuntime: nativeRuntime.validateNativeRuntime
}
