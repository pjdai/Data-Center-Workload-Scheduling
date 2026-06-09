from __future__ import annotations

import numpy as np


def build_delay_matrix(config: dict) -> dict[str, np.ndarray]:
    """Per-task delay penalty over shift offsets s in {0, 1, ..., W}.

    D_s = exp(lambda_c * s / W_c) - 1, so D_0 = 0 and D_W = exp(lambda_c) - 1.
    For W = 0 (no flexibility), returns np.zeros(1) — no penalty applies.
    """
    matrices: dict[str, np.ndarray] = {}
    for task in config["tasks"]:
        name = task["name"]
        w = int(task["flexibility_hours"])
        lam = float(task.get("urgency_lambda", 0.0))
        if w == 0:
            matrices[name] = np.zeros(1)
        else:
            s = np.arange(w + 1, dtype=float)
            matrices[name] = np.exp(lam * s / w) - 1.0
    return matrices
