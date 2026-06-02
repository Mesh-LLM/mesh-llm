'use strict'

const path = require('node:path')
const nativeRuntime = require('./native-runtime')
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

class Client {
  constructor(handle) {
    this._handle = handle
    this.inference = new Inference(handle)
  }

  static create(options) {
    const handle = native.Node.create(
      options.ownerKeypairHex,
      options.inviteToken,
      null,
      null,
      false
    )
    return new Client(handle)
  }

  start() {
    return this._handle.start()
  }

  stop() {
    return this._handle.stop()
  }

  reconnect() {
    return this._handle.reconnect()
  }

  async status() {
    return parse(await this._handle.statusJson())
  }
}

class Console {
  constructor(handle) {
    this._handle = handle
  }

  get url() {
    return this._handle.url
  }

  stop() {
    return this._handle.stop()
  }
}

class Node {
  constructor(handle) {
    this._handle = handle
    this.inference = new Inference(handle)
    this.models = new Models(handle)
    this.serving = new Serving(handle)
  }

  static create(options) {
    const handle = native.Node.create(
      options.ownerKeypairHex,
      options.inviteToken,
      options.cacheDir || null,
      options.runtimeDir || null,
      options.servingEnabled === true
    )
    return new Node(handle)
  }

  start() {
    return this._handle.start()
  }

  stop() {
    return this._handle.stop()
  }

  reconnect() {
    return this._handle.reconnect()
  }

  async status() {
    return parse(await this._handle.statusJson())
  }

  async startConsole(options = {}) {
    const assetDir = options.assetDir || defaultConsoleAssetDir()
    const handle = await this._handle.startConsole(
      assetDir,
      options.port == null ? null : options.port,
      options.listenAll === true
    )
    return new Console(handle)
  }
}

class Inference {
  constructor(handle) {
    this._handle = handle
  }

  async listModels() {
    return parse(await this._handle.listModelsJson())
  }

  async chat(request, options = {}) {
    return parse(await this._handle.chatJson(JSON.stringify(request), options.timeoutMs || null))
  }

  async responses(request, options = {}) {
    return parse(await this._handle.responsesJson(JSON.stringify(request), options.timeoutMs || null))
  }

  cancel(requestId) {
    return this._handle.cancel(requestId)
  }
}

class Models {
  constructor(handle) {
    this._handle = handle
  }

  async recommended() {
    return parse(await this._handle.recommendedModelsJson())
  }

  async search(query) {
    return parse(await this._handle.searchModelsJson(query.query, query.limit || null))
  }

  async show(modelRef) {
    return parse(await this._handle.showModelJson(modelRef))
  }

  async installed() {
    return parse(await this._handle.installedModelsJson())
  }

  async download(modelRef) {
    return parse(await this._handle.downloadModelJson(modelRef))
  }
}

class Serving {
  constructor(handle) {
    this._handle = handle
  }

  async status() {
    return parse(await this._handle.servingStatusJson())
  }

  async load(modelRef, options = {}) {
    return parse(await this._handle.loadServingModelJson(modelRef, JSON.stringify(options)))
  }

  unload(target, options = {}) {
    return this._handle.unloadServingModel(JSON.stringify(target), JSON.stringify(options))
  }

  unloadModel(modelId, options = {}) {
    return this.unload({ modelId }, options)
  }

  unloadInstance(instanceId, options = {}) {
    return this.unload({ instanceId }, options)
  }
}

function parse(json) {
  return JSON.parse(json)
}

function defaultConsoleAssetDir() {
  return path.join(__dirname, 'console')
}

module.exports = {
  Client,
  Console,
  Node,
  generateOwnerKeypairHex: native.generateOwnerKeypairHex,
  currentMeshVersion: nativeRuntime.currentMeshVersion,
  currentSkippyAbiVersion: nativeRuntime.currentSkippyAbiVersion,
  defaultConsoleAssetDir,
  installNativeRuntime: nativeRuntime.installNativeRuntime,
  installedNativeRuntimes: nativeRuntime.installedNativeRuntimes,
  removeNativeRuntime: nativeRuntime.removeNativeRuntime,
  pruneNativeRuntimes: nativeRuntime.pruneNativeRuntimes,
  resolveNativeRuntime: nativeRuntime.resolveNativeRuntime,
  validateNativeRuntime: nativeRuntime.validateNativeRuntime
}
