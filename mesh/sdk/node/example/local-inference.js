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
  const node = Node.create({
    mode: 'combined',
    models: [modelRef],
    joinTokens: process.env.MESH_SDK_INVITE_TOKEN ? [process.env.MESH_SDK_INVITE_TOKEN] : [],
    ownerKeyPath: process.env.MESH_SDK_OWNER_KEY_PATH
  })
  await node.start()
  try {
    const result = await node.inference.chatCompletions({
      model: modelRef,
      messages: [{ role: 'user', content: process.env.MESH_SDK_PROMPT || 'hello' }]
    })
    console.log(result.choices?.[0]?.message?.content)
  } finally {
    await node.stop()
  }
}

main().catch(error => { console.error(error); process.exit(1) })
