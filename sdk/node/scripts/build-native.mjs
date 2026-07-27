import { copyFileSync, mkdirSync } from 'node:fs'
import { basename, join, resolve } from 'node:path'
import { spawnSync } from 'node:child_process'
import { fileURLToPath } from 'node:url'

const repoRoot = resolve(fileURLToPath(new URL('../../..', import.meta.url)))
const profile = process.env.MESH_NODE_PROFILE || 'release'
const cargoArgs = [
  'build',
  '--locked',
  '-p',
  'mesh-llm-nodejs',
  '--no-default-features',
  '--features',
  'embedded-runtime'
]
if (profile === 'release') cargoArgs.push('--release')
if (profile === 'release' && process.platform === 'darwin') {
  // Avoid llvm-objcopy corrupting the LINKEDIT string pool in large Mach-O
  // addons on current macOS runners.
  cargoArgs.push('--config', 'profile.release.strip=false')
}

const build = spawnSync('cargo', cargoArgs, {
  cwd: repoRoot,
  stdio: 'inherit',
  shell: process.platform === 'win32'
})
if (build.status !== 0) {
  process.exit(build.status || 1)
}

const platformArch = `${process.platform}-${process.arch}`
const targetDir = join(repoRoot, 'target', profile)
const source = join(targetDir, nativeLibraryName())
const destinationDir = join(repoRoot, 'sdk', 'node', 'native', platformArch)
mkdirSync(destinationDir, { recursive: true })
copyFileSync(source, join(destinationDir, 'mesh_llm_nodejs.node'))
console.log(`copied ${basename(source)} to ${join(destinationDir, 'mesh_llm_nodejs.node')}`)

function nativeLibraryName() {
  if (process.platform === 'win32') return 'mesh_llm_nodejs.dll'
  if (process.platform === 'darwin') return 'libmesh_llm_nodejs.dylib'
  return 'libmesh_llm_nodejs.so'
}
