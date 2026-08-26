# Release Process

This document describes how maintainers publish a source snapshot with one Git
commit and no inherited repository history. Remote changes require separate
maintainer approval.

## Prepare the release source

1. Record the exact approved release commit as `RELEASE_SHA`.
2. Start from a clean checkout of that commit.
3. Export tracked files into a new directory without copying `.git`:

   ```bash
   git archive "$RELEASE_SHA" | tar -x -C /path/to/new-export
   ```

4. Before initializing Git in the export, verify licenses, source provenance,
   unexpected binary artifacts, credentials, private endpoints, restricted SDK
   material, nested Git metadata, and the complete file manifest.
5. Stop if any finding cannot be classified and cleared.

Exporting before `git init` is important: content that fails the release gate is
never written into the public repository's Git object database.

## Create the single commit

From the verified export:

```bash
git init -b main
git config user.name "<approved-public-name>"
git config user.email "<approved-public-email>"
git add -A
git commit -m "chore: publish TileLang-Sunrise 0.1.13+sunrise.1.0.0"
```

Do not add an existing repository as an alternate and do not fetch the internal
source repository into this new object database.

## Verify history and content

The following checks must all pass:

```bash
test "$(git rev-list --all --count)" -eq 1
test "$(git for-each-ref --format='%(refname)' | wc -l)" -eq 1
test "$(git for-each-ref --format='%(refname)' refs/heads/)" = refs/heads/main
test ! -f .git/shallow
test ! -f .git/objects/info/alternates
test -z "$(git replace -l)"
test -z "$(git ls-files -s | awk '$1 == 160000 {print}')"
test -z "$(find . -path ./.git -prune -o -name .git -print)"
test -z "$(find . -path ./.git -prune -o -name .gitmodules -print)"
git fsck --full --no-reflogs --unreachable
```

Repeat the license, provenance, secret, endpoint, restricted-material, binary,
and file-manifest scans against both the working tree and every Git object.
Then create a fresh local clone of this repository and repeat the history and
content gates in the clone.

## Publish

Only after explicit approval:

1. Verify that the destination repository is empty.
2. Add it as the first remote and push `main` without force.
3. If the destination contains any refs, stop.
4. Fresh-clone the destination and repeat the complete history and content
   verification.
5. Retain the release commit, component revisions, scan logs, license manifest,
   and test results as the publication evidence.
