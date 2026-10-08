#!/usr/bin/env bash
# Deploy the portfolio as a private preview behind a password, on its own Vercel project (mattebell-preview),
# for checking work on other devices (the iPhone) before it ships. The real site is untouched: mattebell.xyz
# deploys from master on its own project, and the gate below exists only in the preview's upload.
#
# Why a script: the Vercel CLI attaches the latest commit to a deploy made inside a git repository, Vercel
# then checks that the commit author is a member of the team and, when it cannot verify that, silently BLOCKS
# the deployment (the CLI hangs on "Building..."). Deploying from a copy without git metadata avoids it.
#
# The password: proxy.ts (copied to the upload's root) lets nobody in until SITE_PASSWORD is set in the
# mattebell-preview project's environment variables (Production). Set it, then deploy again.
#
# Usage:  bash tools/preview/deploy.sh   (from anywhere)
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
BASE="$(mktemp -d)"
TMP="$BASE/mattebell-preview"
mkdir -p "$TMP"
trap 'cd /; rm -rf "$BASE"' EXIT   # (out of the copy first: Windows will not remove the folder you stand in)

# The site's sources. Left behind: git metadata, build output, the site's own Vercel link, design/ (private
# mockups) and tools/ (build and preview scripts).
(cd "$ROOT" && tar --exclude='./.git' --exclude='./node_modules' --exclude='./.next' --exclude='./.vercel' \
  --exclude='./design' --exclude='./tools' --exclude='./*.tsbuildinfo' -cf - .) | (cd "$TMP" && tar -xf -)
if [ -e "$TMP/design" ] || [ -e "$TMP/.git" ]; then echo "deploy.sh: design/ or .git is in the upload. Nothing was deployed." >&2; exit 1; fi

# The gate, and no search engine indexing anywhere in the preview.
cp "$HERE/proxy.ts" "$TMP/proxy.ts"
cat > "$TMP/vercel.json" <<'JSON'
{
  "framework": "nextjs",
  "headers": [{ "source": "/(.*)", "headers": [{ "key": "X-Robots-Tag", "value": "noindex, nofollow" }] }]
}
JSON

cd "$TMP"
# The first time: create the preview project and keep its link here (gitignored).
if [ -f "$HERE/.vercel/project.json" ]; then
  mkdir -p .vercel && cp "$HERE/.vercel/project.json" .vercel/project.json
else
  CI=1 vercel link --yes --project mattebell-preview < /dev/null
  mkdir -p "$HERE/.vercel" && cp .vercel/project.json "$HERE/.vercel/project.json"
fi
CI=1 vercel deploy --prod --yes < /dev/null
