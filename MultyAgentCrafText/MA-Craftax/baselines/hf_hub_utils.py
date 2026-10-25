"""
Optional Hugging Face Hub uploads for MAPPO checkpoints and trajectory datasets.

Upload is off by default. Enable with a repo id plus a token:

  export HF_HUB_REPO_ID=user/mappo-craftext
  export HF_TOKEN=hf_...
  export HF_HUB_UPLOAD=1          # or pass enabled=True / --hf-upload

Without repo id, token, or enable flag, helpers no-op so local runs keep working.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Optional, Sequence, Union


PathLike = Union[str, Path]


def _strip_wrapped_quotes(value: Optional[str]) -> Optional[str]:
    """Drop one pair of matching quotes left by ``docker run --env-file``."""
    if value is None:
        return None
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        value = value[1:-1]
    return value or None


def resolve_hf_token(explicit: Optional[str] = None) -> Optional[str]:
    if explicit:
        return _strip_wrapped_quotes(explicit)
    return _strip_wrapped_quotes(
        os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    )


def resolve_hf_repo_id(explicit: Optional[str] = None) -> Optional[str]:
    if explicit:
        return explicit
    return os.environ.get("HF_HUB_REPO_ID") or os.environ.get("HF_REPO_ID")


def _env_truthy(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def hf_upload_enabled(
    repo_id: Optional[str] = None,
    token: Optional[str] = None,
    enabled: Optional[bool] = None,
) -> bool:
    """
    Upload runs only when all of the following hold:
    - enabled is True, or HF_HUB_UPLOAD / HF_UPLOAD env is truthy
    - a repo id is set (argument or HF_HUB_REPO_ID / HF_REPO_ID)
    - a token is available (argument or HF_TOKEN / HUGGING_FACE_HUB_TOKEN)
    """
    want = bool(enabled) or _env_truthy("HF_HUB_UPLOAD") or _env_truthy("HF_UPLOAD")
    if not want:
        return False

    resolved_repo = resolve_hf_repo_id(repo_id)
    if not resolved_repo:
        print(
            "[hf_hub] Upload enabled but HF_HUB_REPO_ID / HF_REPO_ID is missing; "
            "skipping Hub upload (local files kept)."
        )
        return False

    if not resolve_hf_token(token):
        print(
            "[hf_hub] Upload enabled but HF_TOKEN / HUGGING_FACE_HUB_TOKEN is missing; "
            "skipping Hub upload (local files kept)."
        )
        return False
    return True


def _import_hf_api():
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise ImportError(
            "huggingface_hub is required for Hub upload. "
            "Install with: pip install huggingface_hub"
        ) from exc
    return HfApi


def upload_files_to_hub(
    local_paths: Sequence[PathLike],
    repo_id: Optional[str] = None,
    *,
    token: Optional[str] = None,
    repo_type: str = "model",
    path_in_repo_prefix: str = "",
    private: bool = True,
    create_repo: bool = True,
    enabled: Optional[bool] = None,
) -> list[str]:
    """Upload local files to a Hub repo. Returns Hub paths uploaded (empty if skipped)."""
    resolved_repo = resolve_hf_repo_id(repo_id)
    if not hf_upload_enabled(resolved_repo, token, enabled):
        return []

    HfApi = _import_hf_api()
    api = HfApi(token=resolve_hf_token(token))
    if create_repo:
        api.create_repo(
            repo_id=resolved_repo,
            repo_type=repo_type,
            private=private,
            exist_ok=True,
        )

    uploaded: list[str] = []
    prefix = path_in_repo_prefix.strip("/")
    for raw in local_paths:
        path = Path(raw)
        if not path.is_file():
            print(f"[hf_hub] Skip missing file: {path}")
            continue
        remote_name = f"{prefix}/{path.name}" if prefix else path.name
        api.upload_file(
            path_or_fileobj=str(path),
            path_in_repo=remote_name,
            repo_id=resolved_repo,
            repo_type=repo_type,
        )
        uploaded.append(remote_name)
        print(f"[hf_hub] Uploaded {path} -> {resolved_repo}/{remote_name} ({repo_type})")
    return uploaded


def upload_checkpoint_replacing_previous(
    local_path: PathLike,
    update_step: int,
    repo_id: Optional[str] = None,
    *,
    token: Optional[str] = None,
    repo_type: str = "model",
    path_in_repo_prefix: str = "checkpoints",
    private: bool = True,
    enabled: Optional[bool] = None,
    squash_history: bool = False,
    force: Optional[bool] = None,
) -> list[str]:
    """Upload one checkpoint, replacing the previous file and dropping other names.

    The repository keeps ``mappo_model_latest.pkl`` and ``latest_step.txt``.
    A newer remote step is left in place unless force is set, so a restarted
    run cannot overwrite a longer training checkpoint with step 0.
    """
    resolved_repo = resolve_hf_repo_id(repo_id)
    if not hf_upload_enabled(resolved_repo, token, enabled):
        return []

    local_path = Path(local_path)
    if not local_path.is_file():
        print(f"[hf_hub] Skip missing checkpoint: {local_path}")
        return []

    from huggingface_hub import CommitOperationAdd, CommitOperationDelete

    HfApi = _import_hf_api()
    api = HfApi(token=resolve_hf_token(token))
    api.create_repo(
        repo_id=resolved_repo,
        repo_type=repo_type,
        private=private,
        exist_ok=True,
    )

    prefix = path_in_repo_prefix.strip("/")
    remote_pkl = f"{prefix}/{local_path.name}" if prefix else local_path.name
    remote_step = f"{prefix}/latest_step.txt" if prefix else "latest_step.txt"
    step_file = local_path.with_name("latest_step.txt")
    step_file.write_text(f"{int(update_step)}\n", encoding="utf-8")

    allow_regression = bool(force) or _env_truthy("HF_HUB_FORCE_UPLOAD")
    remote_update_step = _read_remote_step(api, resolved_repo, repo_type, remote_step)
    if (
        remote_update_step is not None
        and int(update_step) < remote_update_step
        and not allow_regression
    ):
        print(
            f"[hf_hub] Remote checkpoint is step {remote_update_step}; "
            f"kept it instead of uploading step {int(update_step)}."
        )
        return []

    try:
        existing = set(api.list_repo_files(repo_id=resolved_repo, repo_type=repo_type))
    except Exception as exc:
        print(f"[hf_hub] Could not list repo files before replace: {exc}")
        existing = set()

    keep = {remote_pkl, remote_step}
    operations = [
        CommitOperationAdd(path_in_repo=remote_pkl, path_or_fileobj=str(local_path)),
        CommitOperationAdd(path_in_repo=remote_step, path_or_fileobj=str(step_file)),
    ]
    removed = []
    for path in sorted(existing):
        if prefix:
            in_checkpoint_dir = path.startswith(f"{prefix}/")
        else:
            in_checkpoint_dir = Path(path).name.startswith("mappo_model_")
        if in_checkpoint_dir and path not in keep:
            operations.append(CommitOperationDelete(path_in_repo=path))
            removed.append(path)

    api.create_commit(
        repo_id=resolved_repo,
        repo_type=repo_type,
        operations=operations,
        commit_message=f"Replace checkpoint at step {int(update_step)}",
    )
    print(
        f"[hf_hub] Replaced checkpoint step {int(update_step)} -> "
        f"{resolved_repo}/{remote_pkl}"
    )
    if removed:
        print(f"[hf_hub] Removed previous repo files: {', '.join(removed)}")

    if squash_history:
        try:
            squash_hub_repo_history(
                resolved_repo,
                token=token,
                repo_type=repo_type,
                enabled=True,
            )
        except Exception as exc:
            print(f"[hf_hub] Checkpoint is uploaded; history squash failed: {exc}")
    return [remote_pkl, remote_step]


def _read_remote_step(api, repo_id: str, repo_type: str, path_in_repo: str) -> Optional[int]:
    try:
        from huggingface_hub import hf_hub_download

        local = hf_hub_download(
            repo_id=repo_id,
            filename=path_in_repo,
            repo_type=repo_type,
            token=api.token,
        )
        return int(Path(local).read_text(encoding="utf-8").strip())
    except Exception:
        return None


def squash_hub_repo_history(
    repo_id: Optional[str] = None,
    *,
    token: Optional[str] = None,
    repo_type: str = "model",
    enabled: Optional[bool] = None,
) -> bool:
    """Replace the Hub repository history with one commit to discard old LFS revisions."""
    resolved_repo = resolve_hf_repo_id(repo_id)
    if not hf_upload_enabled(resolved_repo, token, enabled):
        return False

    HfApi = _import_hf_api()
    api = HfApi(token=resolve_hf_token(token))
    api.super_squash_history(
        repo_id=resolved_repo,
        repo_type=repo_type,
    )
    print(f"[hf_hub] Squashed repository history: {resolved_repo} ({repo_type})")
    return True


def upload_folder_to_hub(
    local_dir: PathLike,
    repo_id: Optional[str] = None,
    *,
    token: Optional[str] = None,
    repo_type: str = "dataset",
    path_in_repo: Optional[str] = None,
    private: bool = True,
    create_repo: bool = True,
    allow_patterns: Optional[Iterable[str]] = None,
    enabled: Optional[bool] = None,
) -> Optional[str]:
    """Upload a local directory to a Hub dataset/model repo. Returns repo id or None."""
    resolved_repo = resolve_hf_repo_id(repo_id)
    if not hf_upload_enabled(resolved_repo, token, enabled):
        return None

    HfApi = _import_hf_api()
    api = HfApi(token=resolve_hf_token(token))
    local_dir = Path(local_dir)
    if not local_dir.is_dir():
        print(f"[hf_hub] Not a directory, skip: {local_dir}")
        return None

    if create_repo:
        api.create_repo(
            repo_id=resolved_repo,
            repo_type=repo_type,
            private=private,
            exist_ok=True,
        )

    kwargs = {}
    if allow_patterns is not None:
        kwargs["allow_patterns"] = list(allow_patterns)

    api.upload_folder(
        folder_path=str(local_dir),
        path_in_repo=path_in_repo,
        repo_id=resolved_repo,
        repo_type=repo_type,
        **kwargs,
    )
    print(f"[hf_hub] Uploaded folder {local_dir} -> {resolved_repo} ({repo_type})")
    return resolved_repo
