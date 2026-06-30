# model/solver.py
"""Checkpoint loading utility shared across training and fine-tuning scripts."""

import torch
import torch.nn as nn


def _fix_state_dict_keys(state: dict) -> dict:
    """Strip a leading 'module.' prefix left over from DataParallel checkpoints."""
    if not isinstance(state, dict):
        return state
    fixed = {}
    for k, v in state.items():
        nk = k[7:] if k.startswith("module.") else k
        fixed[nk] = v
    return fixed


def load_ckpt_safely(model: nn.Module, ckpt_path: str, device: str):
    """
    Robust checkpoint loading for:
      - flat state_dict
      - nested {"model": state_dict} / {"state_dict": state_dict}
      - nested SL_module custom format {"transformer": ..., "score_model": ...}

    Also patches the known mismatched key transformer.pos_alpha when absent
    from the current model definition.
    """
    state = torch.load(ckpt_path, map_location=device)

    # Unwrap common wrappers.
    if isinstance(state, dict):
        for k in ("model", "state_dict", "net", "network"):
            if k in state and isinstance(state[k], dict):
                state = state[k]
                break

    if not isinstance(state, dict):
        raise TypeError(f"Checkpoint is not a dict-like state_dict. type={type(state)}")

    state = _fix_state_dict_keys(state)

    if "transformer.pos_alpha" in state and hasattr(model, "transformer") and not hasattr(model.transformer, "pos_alpha"):
        state.pop("transformer.pos_alpha", None)

    try:
        model.load_state_dict(state, strict=True)
        print("[LOAD] strict=True OK")
        return
    except RuntimeError as e:
        print("[LOAD] strict=True failed:", str(e).split("\n")[0])

    if "transformer.pos_alpha" in state:
        print("[LOAD] removing key transformer.pos_alpha and retrying strict load")
        state.pop("transformer.pos_alpha", None)
        try:
            model.load_state_dict(state, strict=True)
            print("[LOAD] strict=True OK after removing pos_alpha")
            return
        except RuntimeError as e:
            print("[LOAD] strict=True still failed:", str(e).split("\n")[0])

    ret = model.load_state_dict(state, strict=False)
    print("[LOAD] strict=False used. ret =", ret)
