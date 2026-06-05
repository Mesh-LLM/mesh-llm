(() => {
  const link = document.querySelector('[data-github-stars-link]');
  if (!link) return;

  const repo = link.getAttribute('data-github-repo');
  const stars = link.querySelector('[data-github-stars]');
  const release = link.querySelector('[data-github-release]');
  const releaseTargets = document.querySelectorAll('[data-github-release]');
  if (!repo || !stars || !release) return;

  const formatStars = (count) => {
    if (!Number.isFinite(count)) return null;
    if (count < 1000) return new Intl.NumberFormat('en-US').format(count);

    return `${new Intl.NumberFormat('en-US', {
      maximumFractionDigits: count < 10000 ? 1 : 0,
    }).format(count / 1000)}k`;
  };

  const githubFetch = (path) => fetch(`https://api.github.com/repos/${repo}${path}`, {
    headers: { Accept: 'application/vnd.github+json' },
  }).then((response) => (response.ok ? response.json() : null));

  Promise.all([githubFetch(''), githubFetch('/releases/latest')])
    .then(([repoData, releaseData]) => {
      const formattedStars = formatStars(repoData?.stargazers_count);
      const releaseVersion = releaseData?.tag_name;

      if (formattedStars) stars.textContent = formattedStars;
      if (releaseVersion) {
        releaseTargets.forEach((target) => {
          target.textContent = releaseVersion;
          if (target.hasAttribute('data-current-version-label')) {
            target.setAttribute('aria-label', `Current version ${releaseVersion}`);
          }
        });
      }

      link.setAttribute(
        'aria-label',
        `View ${repo} on GitHub, ${stars.textContent} stars, ${release.textContent} release`,
      );
    })
    .catch(() => {
      // Keep the rendered fallback if GitHub is unavailable or rate-limited.
    });
})();
