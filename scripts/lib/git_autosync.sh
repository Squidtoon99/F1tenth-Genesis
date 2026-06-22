#!/usr/bin/env bash
# Shared git commit/push/pull with autonomous merge conflict resolution.
# Sourced by training_supervisor.sh and git_sync_watcher.sh.

GIT_AUTOSYNC_REMOTE="${GIT_REMOTE:-origin}"
GIT_AUTOSYNC_BRANCH="${GIT_BRANCH:-$(git rev-parse --abbrev-ref HEAD)}"

# Paths safe to auto-commit (never outputs/, .env, wandb).
GIT_AUTOSYNC_PATHS=(
  scripts/
  standalone_trainer.py
  config.py
  f1tenth_env/
  docs/
)

_git_autosync_author() {
  git log -1 --format='%an|%ae' 2>/dev/null || echo "Training Supervisor|supervisor@local"
}

_git_autosync_log() {
  local log_file="$1"
  shift
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] git_autosync $*" >>"$log_file"
}

fix_script_crlf() {
  local log_file="${1:-/dev/null}"
  local f
  for f in scripts/*.sh; do
    [ -f "$f" ] || continue
    if grep -q $'\r' "$f" 2>/dev/null; then
      sed -i 's/\r$//' "$f"
      _git_autosync_log "$log_file" "fixed CRLF in $f"
    fi
  done
  if [ -d scripts/lib ]; then
    for f in scripts/lib/*.sh; do
      [ -f "$f" ] || continue
      if grep -q $'\r' "$f" 2>/dev/null; then
        sed -i 's/\r$//' "$f"
        _git_autosync_log "$log_file" "fixed CRLF in $f"
      fi
    done
  fi
}

safe_git_commit() {
  local log_file="${1:-/dev/null}"
  local author email name
  IFS='|' read -r name email < <(_git_autosync_author)

  local path has_changes=false
  for path in "${GIT_AUTOSYNC_PATHS[@]}"; do
    if git status --porcelain -- "$path" 2>/dev/null | grep -q .; then
      has_changes=true
      git add -- "$path"
    fi
  done

  if [ "$has_changes" = false ]; then
    return 1
  fi

  GIT_AUTHOR_NAME="$name" GIT_AUTHOR_EMAIL="$email" \
  GIT_COMMITTER_NAME="$name" GIT_COMMITTER_EMAIL="$email" \
    git commit -m "Supervisor autosave $(date '+%Y-%m-%d %H:%M:%S')" >>"$log_file" 2>&1
  _git_autosync_log "$log_file" "committed local changes"
  return 0
}

# Resolve merge conflicts: prefer remote for training code, local for supervisor files.
_resolve_merge_conflicts() {
  local log_file="$1"
  local f
  local conflicted
  conflicted=$(git diff --name-only --diff-filter=U 2>/dev/null || true)
  if [ -z "$conflicted" ]; then
    return 0
  fi

  _git_autosync_log "$log_file" "resolving conflicts in: $conflicted"
  for f in $conflicted; do
    if [[ "$f" == scripts/training_supervisor* ]] \
      || [[ "$f" == scripts/lib/* ]] \
      || [[ "$f" == scripts/start_supervisor* ]]; then
      git checkout --ours -- "$f" 2>/dev/null || true
    else
      git checkout --theirs -- "$f" 2>/dev/null || git checkout --ours -- "$f" 2>/dev/null || true
    fi
    git add -- "$f"
  done

  local author email name
  IFS='|' read -r name email < <(_git_autosync_author)
  GIT_AUTHOR_NAME="$name" GIT_AUTHOR_EMAIL="$email" \
  GIT_COMMITTER_NAME="$name" GIT_COMMITTER_EMAIL="$email" \
    git commit -m "Supervisor: auto-resolved merge conflicts" >>"$log_file" 2>&1 \
    || git merge --continue >>"$log_file" 2>&1 \
    || git rebase --continue >>"$log_file" 2>&1 \
    || return 1
  _git_autosync_log "$log_file" "conflicts resolved"
  return 0
}

safe_git_sync() {
  local log_file="${1:-/dev/null}"
  local remote="$GIT_AUTOSYNC_REMOTE"
  local branch="$GIT_AUTOSYNC_BRANCH"

  if ! git fetch "$remote" "$branch" >>"$log_file" 2>&1; then
    _git_autosync_log "$log_file" "fetch failed"
    return 1
  fi

  local local_sha remote_sha ahead behind
  local_sha=$(git rev-parse HEAD)
  remote_sha=$(git rev-parse "${remote}/${branch}" 2>/dev/null || echo "")

  if [ -z "$remote_sha" ]; then
    _git_autosync_log "$log_file" "no remote ref ${remote}/${branch}"
    return 1
  fi

  ahead=$(git rev-list --count "${remote_sha}..HEAD" 2>/dev/null || echo 0)
  behind=$(git rev-list --count "HEAD..${remote_sha}" 2>/dev/null || echo 0)

  _git_autosync_log "$log_file" "head=${local_sha:0:8} ahead=$ahead behind=$behind"

  if [ "$behind" -gt 0 ] && [ "$ahead" -eq 0 ]; then
    if git pull --ff-only "$remote" "$branch" >>"$log_file" 2>&1; then
      _git_autosync_log "$log_file" "fast-forward pull ok"
    else
      _git_autosync_log "$log_file" "ff-only pull failed; trying merge"
      git merge "${remote}/${branch}" -m "Supervisor merge" >>"$log_file" 2>&1 \
        || _resolve_merge_conflicts "$log_file" \
        || { git merge --abort >>"$log_file" 2>&1; return 1; }
    fi
  elif [ "$behind" -gt 0 ] && [ "$ahead" -gt 0 ]; then
    _git_autosync_log "$log_file" "diverged; merging remote"
    if ! git merge "${remote}/${branch}" -m "Supervisor merge diverged" >>"$log_file" 2>&1; then
      _resolve_merge_conflicts "$log_file" || {
        git merge --abort >>"$log_file" 2>&1
        _git_autosync_log "$log_file" "merge failed; aborting"
        return 1
      }
    fi
  fi

  ahead=$(git rev-list --count "${remote}/${branch}..HEAD" 2>/dev/null || echo 0)
  if [ "$ahead" -gt 0 ]; then
    if git push "$remote" "$branch" >>"$log_file" 2>&1; then
      _git_autosync_log "$log_file" "pushed $ahead commit(s)"
    else
      _git_autosync_log "$log_file" "push failed; pulling and retrying"
      git pull --no-rebase "$remote" "$branch" >>"$log_file" 2>&1 \
        || _resolve_merge_conflicts "$log_file" \
        || { git merge --abort >>"$log_file" 2>&1; return 1; }
      git push "$remote" "$branch" >>"$log_file" 2>&1 \
        || { _git_autosync_log "$log_file" "push retry failed"; return 1; }
    fi
  fi

  return 0
}
