import pathlib
import sys

import yaml


def main() -> int:
    repo_root = pathlib.Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    config_path = repo_root / "configs" / "default.yaml"

    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    required_keys = {"project", "classes", "signal", "features", "paths"}
    missing = required_keys.difference(cfg.keys())
    if missing:
        print(f"Missing config sections: {sorted(missing)}")
        return 1

    # Validate canonical sampling assumptions from tasks.md
    if cfg["signal"]["fs_hz"] != 5000 or cfg["signal"]["samples_per_frame"] != 500:
        print("Config sampling assumptions do not match fs=5000 and N=500")
        return 1

    # Import checks from project root package layout
    import src  # noqa: F401
    import src.io  # noqa: F401
    import src.dsp  # noqa: F401
    import src.data  # noqa: F401
    import src.models  # noqa: F401
    import src.train  # noqa: F401
    import src.eval  # noqa: F401
    import src.adapt  # noqa: F401
    import src.infer  # noqa: F401
    import src.ui  # noqa: F401
    import src.runtime  # noqa: F401
    import src.system  # noqa: F401

    print("Smoke test passed: config loaded and src imports resolved.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
