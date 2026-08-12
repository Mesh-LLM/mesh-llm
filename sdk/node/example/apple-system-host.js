'use strict'

const fs = require('node:fs')
const {
  ProviderHost,
  packagedAppleSystemProvider
} = require('..')

async function main() {
  const rootDir = process.argv[2]
  const readyFile = process.argv[3]
  const stopFile = process.argv[4]
  if (!rootDir || !readyFile || !stopFile) {
    throw new Error('usage: apple-system-host <provider-root> <ready-file> <stop-file>')
  }

  const host = await ProviderHost.start(packagedAppleSystemProvider({ rootDir }))
  fs.writeFileSync(readyFile, JSON.stringify({
    carrier: 'node-electron',
    apiBaseUrl: host.apiBaseUrl
  }))

  while (!fs.existsSync(stopFile)) {
    await new Promise((resolve) => setTimeout(resolve, 100))
  }
  await host.stop()
}

main().catch((error) => {
  console.error(error)
  process.exit(1)
})
