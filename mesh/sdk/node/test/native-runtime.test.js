'use strict'

const test = require('node:test')
const assert = require('node:assert/strict')
const runtime = require('../native-runtime')

test('installNativeRuntime delegates to native SDK installer', async () => {
  const calls = []
  const progress = []
  runtime.configureNativeRuntimeBinding({
    installNativeRuntimeJson: async (json, onProgress) => {
      calls.push(JSON.parse(json))
      onProgress?.(JSON.stringify({
        nativeRuntimeId: 'meshllm-native-runtime-linux-x86_64-cpu',
        url: 'https://example.invalid/runtime.tar.gz',
        downloadedBytes: 128,
        totalBytes: 256,
        finished: false
      }))
      return JSON.stringify({
        status: 'installed',
        runtime: {
          meshVersion: '0.68.0',
          nativeRuntimeId: 'meshllm-native-runtime-linux-x86_64-cpu',
          flavor: 'cpu',
          path: '/cache/runtime',
          skippyAbiVersion: '1.2.3'
        },
        selectedNativeRuntimeId: 'meshllm-native-runtime-linux-x86_64-cpu',
        selectedSource: 'download'
      })
    }
  })

  const outcome = await runtime.installNativeRuntime({
    selection: 'cpu',
    cacheDir: '/cache',
    allowDownload: false,
    onProgress: (event) => progress.push(event)
  })

  assert.equal(calls[0].selection, 'cpu')
  assert.equal(calls[0].cacheDir, '/cache')
  assert.equal(calls[0].allowDownload, false)
  assert.equal(outcome.runtime.nativeRuntimeId, 'meshllm-native-runtime-linux-x86_64-cpu')
  assert.equal(progress[0].downloadedBytes, 128)
})

test('validateNativeRuntime resolves a bundled runtime without download', async () => {
  let installOptions = null
  runtime.configureNativeRuntimeBinding({
    installNativeRuntimeJson: async (json) => {
      installOptions = JSON.parse(json)
      return JSON.stringify({
        status: 'installed',
        runtime: {
          meshVersion: '0.68.0',
          nativeRuntimeId: 'meshllm-native-runtime-darwin-aarch64-metal',
          flavor: 'metal',
          path: '/cache/runtime',
          skippyAbiVersion: '1.2.3'
        },
        selectedNativeRuntimeId: 'meshllm-native-runtime-darwin-aarch64-metal',
        selectedSource: 'bundle'
      })
    }
  })

  const installed = await runtime.validateNativeRuntime('/app/native')

  assert.deepEqual(installOptions.bundleDirs, ['/app/native'])
  assert.equal(installOptions.allowDownload, false)
  assert.equal(installed.flavor, 'metal')
})
