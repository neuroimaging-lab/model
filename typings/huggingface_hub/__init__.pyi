from __future__ import annotations

from pathlib import Path
from typing import Optional

def hf_hub_download(
    *,
    repo_id: str,
    filename: str,
    revision: Optional[str] = ...,
    force_download: bool = ...,
    local_files_only: bool = ...,
    cache_dir: Optional[str | Path] = ...,
) -> str: ...
