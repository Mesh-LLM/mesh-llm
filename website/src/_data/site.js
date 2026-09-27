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
  // Last-resort values only. Both are replaced by the live GitHub values below
  // when the build has network access, and src/assets/github-stars.js refreshes
  // them again in the browser. Only an offline build that is also served to a
  // client that cannot reach api.github.com renders these.
  githubStarsFallback: '3.5k',
  githubReleaseFallback: `v${sdkVersion}`,
  sdkVersion,
};

const GITHUB_TIMEOUT_MS = 5000;

const fetchGithubJson = async (path, describe) => {
  try {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), GITHUB_TIMEOUT_MS);

    const response = await fetch(`https://api.github.com/repos/${site.githubRepo}${path}`, {
      headers: {
        Accept: 'application/vnd.github+json',
        'User-Agent': 'mesh-llm-website',
      },
      signal: controller.signal,
    });

    clearTimeout(timeoutId);
    if (!response.ok) {
      console.warn(`GitHub API returned ${response.status} for ${path}; using the committed fallback ${describe}`);
      return null;
    }

    return await response.json();
  } catch (err) {
    console.warn(`Failed to fetch GitHub ${describe}, falling back to the committed value:`, err);
    return null;
  }
};

const fetchLatestReleaseTag = async () => {
  const release = await fetchGithubJson('/releases/latest', 'release version');
  const tagName = typeof release?.tag_name === 'string' ? release.tag_name.trim() : '';
  return tagName || null;
};

// The star count changes without the repository changing, so it is fetched at
// build time instead of being written into the pages. Keep this in step with
// formatStars() in src/assets/github-stars.js, which formats the same count in
// the browser after the page loads.
const formatStarCount = (count) => {
  if (!Number.isFinite(count)) return null;
  if (count < 1000) return new Intl.NumberFormat('en-US').format(count);

  return `${new Intl.NumberFormat('en-US', {
    maximumFractionDigits: count < 10000 ? 1 : 0,
  }).format(count / 1000)}k`;
};

const fetchStargazerCount = async () => {
  const repo = await fetchGithubJson('', 'star count');
  const count = Number(repo?.stargazers_count);
  return Number.isFinite(count) ? count : null;
};

export default async function () {
  const [releaseTag, stargazers] = await Promise.all([
    fetchLatestReleaseTag(),
    fetchStargazerCount(),
  ]);

  return {
    ...site,
    githubReleaseFallback: releaseTag ?? site.githubReleaseFallback,
    githubStarsFallback: formatStarCount(stargazers) ?? site.githubStarsFallback,
  };
}
