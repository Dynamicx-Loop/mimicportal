# PartyMimic upstream

MimicPortal started from [Roboparty/PartyMimic](https://github.com/Roboparty/PartyMimic), default branch `main`, at commit `b67af29051bcb5aba6d06148c188823a455fcb27`. The original MIT license and copyright notices are retained in [LICENSE](LICENSE). PartyMimic credits BlinkMimic and InstantTrack in that license.

`main` begins with a new root commit containing that snapshot. `partymimic` retains the upstream history and is reserved for an unmodified mirror of upstream `main`. The two branches have unrelated histories; do not merge them directly. Develop on `main` or feature branches based on it.

## Configure and update the mirror

Git remotes are local to each clone. Configure `upstream` once, then fetch and fast-forward the mirror:

```sh
git remote add upstream https://github.com/Roboparty/PartyMimic.git
git fetch upstream main
git switch partymimic
git merge --ff-only upstream/main
git push origin partymimic
```

If the remote already exists, check it with `git remote -v` instead of adding it again. If the fast-forward fails, investigate a rewritten upstream history or local commits; do not force-push the mirror without reviewing the consequences. Fetching and updating the mirror does not change `main`.

## Bring upstream changes into development

The last upstream snapshot incorporated into `main` is initially `b67af29051bcb5aba6d06148c188823a455fcb27`. After updating the mirror, review the changes since that commit. On a clean feature branch based on `main`, apply a binary-capable diff and resolve any conflicts:

```sh
git switch main
git pull --ff-only origin main
git switch -c integrate-partymimic
git diff --stat b67af29051bcb5aba6d06148c188823a455fcb27 partymimic
git diff --binary b67af29051bcb5aba6d06148c188823a455fcb27 partymimic > /tmp/partymimic-update.patch
git apply --3way --index /tmp/partymimic-update.patch
```

Review the staged changes, resolve conflicts if any, check upstream license and attribution changes, and run relevant tests before committing and proposing the integration. Record the exact `partymimic` commit integrated in this file and in the integration commit message; use that commit as the base of the next diff. Applying a snapshot diff does not preserve individual upstream commits on `main`; the original commits remain available on `partymimic`. If the branches have diverged substantially, selectively port changes instead.
