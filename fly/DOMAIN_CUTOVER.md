# Domain cutover runbook — meshllm.cloud

Goal: reshuffle the three public hostnames without breaking live inference or
the public site.

| Hostname | Today | Target |
|---|---|---|
| `meshllm.cloud` (apex) | Console (Fly) | **Marketing site** (GitHub Pages) |
| `docs.meshllm.cloud` | Marketing + docs (GH Pages via Cloudflare) | **301 redirect → `meshllm.cloud/docs/`** |
| `public.meshllm.cloud` | — | **Console** (Fly) ✅ done |

Three control planes are involved:

- **Fly** — runs the console app `mesh-llm-console` (`fly/console/fly.toml`).
- **GitHub Pages** — builds/serves the static site from `website/` via
  `.github/workflows/website-pages.yml`; custom domain driven by
  `website/src/CNAME`. Pages origin host for this repo is `mesh-llm.github.io`.
- **Cloudflare** — DNS + proxy for the `meshllm.cloud` zone.

---

## Status

- [x] **Step 1 — Console on `public.meshllm.cloud`** (DONE)
  - `fly certs add public.meshllm.cloud -a mesh-llm-console` — issued + verified.
  - Cloudflare DNS: `CNAME public -> mesh-llm-console.fly.dev`, **DNS only (grey)**.
  - `fly/console/fly.toml` + `fly/Dockerfile`: `VITE_API_URL=https://public.meshllm.cloud`.
  - Console redeployed; `https://public.meshllm.cloud/api/status` + `/v1/models` verified.
- [x] **Repo edits for apex** (DONE, committed on branch `console-public-domain`)
  - `website/src/CNAME -> meshllm.cloud`
  - `website/src/_data/site.js` canonical `url -> https://meshllm.cloud`
  - `website/src/funding.json` + `.well-known/funding-manifest-urls -> apex`
  - `fly/README.md` updated.
- [ ] **Step 2 — Flip apex to GitHub Pages** (BELOW)
- [ ] **Step 3 — `docs.` redirect** (BELOW)
- [ ] **Step 4 — Decommission apex on Fly** (BELOW)
- [ ] **Step 5 — Follow-up: UI anchor links** (BELOW, no urgency)

---

## Pre-flight

Record current apex DNS so rollback is instant:

```
meshllm.cloud  A  66.241.124.167   (Fly)   <-- rollback target
```

Merge branch `console-public-domain` to `main` before triggering the Pages
build (the build deploys from `main`). The console redeploy already shipped; the
website CNAME change only takes effect once Pages rebuilds from `main`.

---

## Step 2 — Flip apex `meshllm.cloud` to the marketing site

Order matters: set the Pages custom domain first, then DNS, then build.

1. **GitHub → repo Settings → Pages**
   - Set **Custom domain** to `meshllm.cloud`. Save.
   - Leave **Enforce HTTPS** unchecked until the cert provisions (GitHub issues
     it after DNS resolves to Pages), then enable it.

2. **Cloudflare → DNS** for the `meshllm.cloud` zone — repoint the **apex**:
   - Remove the existing apex `A 66.241.124.167` (Fly).
   - Add the four GitHub Pages A records (Name = `@` / `meshllm.cloud`),
     **Proxied (orange) is fine** with Cloudflare:
     ```
     A  @  185.199.108.153
     A  @  185.199.109.153
     A  @  185.199.110.153
     A  @  185.199.111.153
     ```
   - (Optional IPv6) AAAA records:
     ```
     AAAA @ 2606:50c0:8000::153
     AAAA @ 2606:50c0:8001::153
     AAAA @ 2606:50c0:8002::153
     AAAA @ 2606:50c0:8003::153
     ```
   - Alternative: a single Cloudflare CNAME-flattened `@ -> mesh-llm.github.io`
     (Cloudflare flattens apex CNAMEs automatically). Pick A-records OR the
     flattened CNAME, not both.

3. **Trigger the Pages build** so `CNAME` = `meshllm.cloud` is published:
   - Merge `console-public-domain` to `main` (touches `website/**`), or
   - Run the **Public Website Deploy** workflow via `workflow_dispatch` on `main`.

4. **Verify**
   ```bash
   dig +short meshllm.cloud            # expect 185.199.108-111.153 (or CF proxy IPs)
   curl -sI https://meshllm.cloud/      | grep -i 'server\|content-type'  # GH Pages, text/html
   curl -s  https://meshllm.cloud/docs/ | grep -i '<title>'               # Docs title
   ```

**Rollback:** in Cloudflare, restore apex `A 66.241.124.167`. Apex serves the
console again within DNS TTL.

---

## Step 3 — `docs.meshllm.cloud` → 301 redirect to `meshllm.cloud/docs/`

`docs.meshllm.cloud` is no longer a Pages custom domain (Pages allows one, now
the apex). Keep the name alive as a redirect so old links never die.

1. **Cloudflare → DNS** — make `docs` a **proxied** placeholder so the redirect
   rule can run (must be **orange cloud**):
   ```
   CNAME  docs  meshllm.cloud   (Proxied / orange)
   ```
   (If a `docs` record already exists pointing at GitHub Pages, just switch it to
   this proxied CNAME.)

2. **Cloudflare → Rules → Redirect Rules → Create rule**
   - **When incoming requests match:** `Hostname equals docs.meshllm.cloud`
   - **Then → Dynamic redirect:**
     - Status code: **301 (Permanent)**
     - Target expression:
       ```
       concat("https://meshllm.cloud/docs", http.request.uri.path)
       ```
     - **Preserve query string:** on

3. **Verify** (path-preserving):
   ```bash
   curl -sI https://docs.meshllm.cloud/         # 301 -> https://meshllm.cloud/docs/
   curl -sI https://docs.meshllm.cloud/install  # 301 -> https://meshllm.cloud/docs/install
   ```

Note: `#fragment` anchors (e.g. `docs.meshllm.cloud/#install`) redirect to
`meshllm.cloud/docs/#install`. The page loads; the anchor is wrong because
`#install` lives on the marketing home (`meshllm.cloud/#install`). Step 5 fixes
the source links.

---

## Step 4 — Decommission the apex on Fly

Once apex serves Pages and is verified, stop Fly from claiming `meshllm.cloud`:

```bash
fly certs remove meshllm.cloud -a mesh-llm-console
# Keep public.meshllm.cloud and *.fly.dev certs.
```

Verify the console is unaffected:
```bash
curl -s https://public.meshllm.cloud/api/status | head -c 120
```

---

## Step 5 — Follow-up: fix in-console UI links (no urgency)

These are baked into the console UI bundle and require a console rebuild +
redeploy. Not blocking; covered by the Step 3 redirect in the meantime.

In `crates/mesh-llm-ui/src/`:

- Keep links meant for **docs** as `https://docs.meshllm.cloud/` (redirects to
  `/docs/`) — or update to `https://meshllm.cloud/docs/` directly.
- Fix **marketing-home anchors** to the apex:
  - `https://docs.meshllm.cloud/#install` -> `https://meshllm.cloud/#install`
  - `https://docs.meshllm.cloud/#blackboard` -> `https://meshllm.cloud/#blackboard`
- Files with these refs: `features/app-tabs/data.ts`,
  `features/shell/components/TopNav.tsx`, `features/app-shell/components/AppHeader.tsx`,
  `features/chat/components/ChatPage.tsx`,
  `features/configuration/pages/ConfigurationPage.tsx`,
  `constants/default-system-prompt.md`, plus their `.test.tsx` expectations.

After editing, rebuild + redeploy the console:
```bash
fly deploy --config fly/console/fly.toml --dockerfile fly/Dockerfile
```

---

## Quick reference — final DNS state (Cloudflare)

| Type | Name | Target | Proxy |
|---|---|---|---|
| A | `@` | 185.199.108–111.153 (GH Pages) | proxied or DNS-only |
| CNAME | `public` | `mesh-llm-console.fly.dev` | **DNS only (grey)** |
| CNAME | `docs` | `meshllm.cloud` | **Proxied (orange)** + redirect rule |
