'use strict'

const { Node, resolveNativeRuntime } = require('..')

async function main() {
  const modelRef = process.env.MESH_SDK_MODEL_REF
  if (!modelRef) throw new Error('Set MESH_SDK_MODEL_REF')
  const runtime = await resolveNativeRuntime({
    artifactDir: process.env.MESHLLM_NATIVE_RUNTIME_ARTIFACT_DIR,
    allowDownload: process.env.MESH_SDK_RUNTIME_ALLOW_DOWNLOAD === '1'
  })
  console.log(`using native runtime ${runtime.nativeRuntimeId}`)
  if (process.env.MESHLLM_NATIVE_RUNTIME_ARTIFACT_DIR) {
    process.env.MESH_LLM_NATIVE_RUNTIME_BUNDLE_DIR = process.env.MESHLLM_NATIVE_RUNTIME_ARTIFACT_DIR
  }
  const node = Node.create({
    mode: 'combined',
    models: [modelRef],
    joinTokens: process.env.MESH_SDK_INVITE_TOKEN ? [process.env.MESH_SDK_INVITE_TOKEN] : [],
    ownerKeyPath: process.env.MESH_SDK_OWNER_KEY_PATH
  })
  await node.start()
  try {
    const requestedModel = modelRef.split(/[\\/]/).pop().replace(/\.gguf$/i, '')
    let model
    for (let attempt = 0; attempt < 120; attempt++) {
      const models = await node.inference.listModels()
      model = models.find(item => item.id === modelRef || item.id === requestedModel)
      if (model) break
      await new Promise(resolve => setTimeout(resolve, 1000))
    }
    if (!model) throw new Error(`Model ${modelRef} did not become available within two minutes`)
    const result = await node.inference.chatCompletions({
      model: model.id,
      messages: [{ role: 'user', content: process.env.MESH_SDK_PROMPT || 'hello' }]
    })
    console.log(result.choices?.[0]?.message?.content)
  } finally {
    await node.stop()
  }
}

main().catch(error => { console.error(error); process.exit(1) })
