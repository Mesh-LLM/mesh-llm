'use strict'

const { Node, generateOwnerKeypairHex, resolveNativeRuntime } = require('..')

async function main() {
  const modelRef = process.env.MESH_SDK_MODEL_REF
  const inviteToken = process.argv[2] || process.env.MESH_SDK_INVITE_TOKEN || 'local'
  if (!modelRef) {
    console.error('Set MESH_SDK_MODEL_REF to run local serving.')
    process.exit(2)
  }

  const runtime = await resolveNativeRuntime({
    artifactDir: process.env.MESHLLM_NATIVE_RUNTIME_ARTIFACT_DIR,
    allowDownload: process.env.MESH_SDK_RUNTIME_ALLOW_DOWNLOAD === '1'
  })
  console.log(`using native runtime ${runtime.nativeRuntimeId} from ${runtime.path}`)

  const node = Node.create({
    ownerKeypairHex: process.env.MESH_SDK_OWNER_KEYPAIR_HEX || generateOwnerKeypairHex(),
    inviteToken,
    cacheDir: process.env.MESH_SDK_CACHE_DIR,
    runtimeDir: process.env.MESH_SDK_RUNTIME_DIR,
    servingEnabled: true
  })

  await node.start()
  try {
    if (process.env.MESH_SDK_SKIP_DOWNLOAD !== '1') {
      await node.models.download(modelRef)
    }
    const served = await node.serving.load(modelRef, { devicePolicy: 'auto' })
    const result = await node.inference.chat({
      model: served.modelId,
      messages: [
        { role: 'user', content: process.env.MESH_SDK_PROMPT || 'hello' }
      ]
    })
    console.log(result.content)
    if (served.instanceId) {
      await node.serving.unloadInstance(served.instanceId, { drainTimeoutMs: 1000 })
    } else {
      await node.serving.unloadModel(served.modelId, { drainTimeoutMs: 1000 })
    }
  } finally {
    await node.stop()
  }
}

main().catch((error) => {
  console.error(error)
  process.exit(1)
})
