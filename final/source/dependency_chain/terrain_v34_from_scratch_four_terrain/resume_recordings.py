"""Resume V34 recordings after all fixed evaluations have completed."""
import json
import os
from pathlib import Path
import subprocess

import supervisor as common
from catalog import CASES


def main():
    batch = Path(os.environ["V34_RECOVERY_BATCH"]).resolve()
    model = Path(os.environ["V34_RECOVERY_CHECKPOINT"]).resolve()
    flat_model = Path(os.environ["V34_RECOVERY_FLAT_CHECKPOINT"]).resolve()
    if not batch.is_dir() or not model.is_file() or not flat_model.is_file():
        raise ValueError("Invalid V34 recording recovery paths")
    common.BATCH = batch
    results = {}
    for case in CASES:
        result_path = batch / f"V34_{case}.json"
        if not result_path.is_file():
            raise FileNotFoundError(result_path)
        results[case] = json.loads(result_path.read_text())
    for case, (_, difficulty) in CASES.items():
        common.record(model, case, difficulty, f"V34_{case}")
        common.publish(results, flat_model, model)
    page = common.publish(results, flat_model, model)
    subprocess.Popen(["firefox", "--new-window", page.as_uri()], env=common.ENV)
    common.status("complete", checkpoint=str(model), flat_checkpoint=str(flat_model), page=str(page), recovered_recordings=True)


if __name__ == "__main__":
    main()
