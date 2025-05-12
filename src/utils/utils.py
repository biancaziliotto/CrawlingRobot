from pathlib import Path

def get_last_ckpt(dir_path: Path, mode: str = "backward") -> Path | None:
    """Return the checkpoint with the highest numeric suffix, or None if none exist."""
    return max(
        dir_path.glob(f"model_{mode}_*.ckpt"),
        key=lambda p: int(p.stem.split('_')[-1]),
        default=None,
    )