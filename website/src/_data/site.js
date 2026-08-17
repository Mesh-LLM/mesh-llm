import { readFileSync } from 'node:fs';

const cargoToml = readFileSync(new URL('../../../Cargo.toml', import.meta.url), 'utf8');
const sdkVersion = cargoToml.match(
  /\[workspace\.package\][\s\S]*?\nversion\s*=\s*"([^"]+)"/
)?.[1];

if (!sdkVersion) {
  throw new Error('Unable to determine the SDK version from Cargo.toml');
}

const site = {
  title: 'Mesh LLM',
  description: 'Mesh serves large local models across multiple machines through one OpenAI-compatible endpoint.',
  url: 'https://meshllm.cloud',
  publicMeshUrl: 'https://public.meshllm.cloud',
  githubUrl: 'https://github.com/Mesh-LLM/mesh-llm',
  githubRepo: 'Mesh-LLM/mesh-llm',
  githubStarsFallback: '1.1k',
  githubReleaseFallback: `v${sdkVersion}`,
  sdkVersion,
};

const fetchLatestReleaseTag = async (repo) => {
  try {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 5000);

    const response = await fetch(`https://api.github.com/repos/${repo}/releases/latest`, {
      headers: {
        Accept: 'application/vnd.github+json',
        'User-Agent': 'mesh-llm-website',
      },
      signal: controller.signal,
    });

    clearTimeout(timeoutId);
    if (!response.ok) {
      console.warn(`GitHub API returned ${response.status}; using fallback release version`);
      return null;
    }

    const release = await response.json();
    const tagName = typeof release?.tag_name === 'string' ? release.tag_name.trim() : '';
    return tagName || null;
  } catch (err) {
    console.warn('Failed to fetch GitHub release, falling back to hardcoded version:', err);
    return null;
  }
};

export default async function () {
  const githubReleaseFallback = await fetchLatestReleaseTag(site.githubRepo) ?? site.githubReleaseFallback;

  return {
    ...site,
    githubReleaseFallback,
  };
}
