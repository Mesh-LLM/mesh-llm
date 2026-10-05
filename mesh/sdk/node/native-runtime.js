'use strict'

let nativeBinding = null

function configureNativeRuntimeBinding(binding) {
  nativeBinding = binding
}

function currentMeshVersion() {
  return native().currentMeshVersion()
}

function currentSkippyAbiVersion() {
  return native().currentSkippyAbiVersion()
}

async function installNativeRuntime(options = {}) {
  const onProgress = typeof options.onProgress === 'function'
    ? (eventJson) => options.onProgress(parse(eventJson))
    : null
  return parse(await native().installNativeRuntimeJson(
    JSON.stringify(normalizeInstallOptions(options)),
    onProgress
  ))
}

async function installedNativeRuntimes(options = {}) {
  return parse(await native().installedNativeRuntimesJson(options.cacheDir || null))
}

function removeNativeRuntime(options) {
  if (!options || !options.meshVersion || !options.nativeRuntimeId) {
    throw new Error('removeNativeRuntime requires meshVersion and nativeRuntimeId')
  }
  return native().removeNativeRuntime(
    options.cacheDir || null,
    options.meshVersion,
    options.nativeRuntimeId
  )
}

function pruneNativeRuntimes(options = {}) {
  return parse(native().pruneNativeRuntimesJson(
    options.cacheDir || null,
    options.activeMeshVersion || null,
    options.mode || null
  ))
}

async function resolveNativeRuntime(options = {}) {
  const bundleDirs = []
  if (options.artifactDir) bundleDirs.push(options.artifactDir)
  for (const dir of options.searchDirs || []) bundleDirs.push(dir)
  const outcome = await installNativeRuntime({
    ...options,
    bundleDirs,
    allowDownload: options.allowDownload === true
  })
  return outcome.runtime
}

async function validateNativeRuntime(artifactDir, options = {}) {
  const outcome = await installNativeRuntime({
    ...options,
    bundleDirs: [artifactDir],
    allowDownload: false
  })
  return outcome.runtime
}

function normalizeInstallOptions(options) {
  return {
    meshVersion: options.meshVersion || null,
    skippyAbiVersion: options.skippyAbiVersion || null,
    selection: options.selection || 'recommended',
    manifestPath: options.manifestPath || null,
    manifestUrl: options.manifestUrl || null,
    bundleDirs: options.bundleDirs || [],
    cacheDir: options.cacheDir || null,
    verificationPolicy: options.verificationPolicy || 'require_checksum',
    allowDownload: options.allowDownload !== false
  }
}

function native() {
  if (!nativeBinding) {
    throw new Error('MeshLLM native runtime binding has not been configured')
  }
  return nativeBinding
}

function parse(json) {
  return JSON.parse(json)
}

module.exports = {
  configureNativeRuntimeBinding,
  currentMeshVersion,
  currentSkippyAbiVersion,
  installNativeRuntime,
  installedNativeRuntimes,
  removeNativeRuntime,
  pruneNativeRuntimes,
  resolveNativeRuntime,
  validateNativeRuntime
}
