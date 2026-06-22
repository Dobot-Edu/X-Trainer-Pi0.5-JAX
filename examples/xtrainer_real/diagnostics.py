import logging
from typing import Any

import numpy as np
from openpi_client.runtime import subscriber as _subscriber
from typing_extensions import override

logger = logging.getLogger(__name__)


def _vec_line(name: str, values: np.ndarray, digits: int = 3) -> str:
    vec = np.asarray(values).reshape(-1)
    return f"{name}[{vec.shape[0]}]=" + np.array2string(vec, precision=digits, suppress_small=False, max_line_width=240)


class ActionStateDiagnosticsSubscriber(_subscriber.Subscriber):
    """Read-only runtime diagnostics for action/state mismatch analysis."""

    def __init__(
        self,
        *,
        interval: int = 20,
        max_steps: int = 400,
        eps: float = 1e-3,
    ) -> None:
        self._interval = max(int(interval), 1)
        self._max_steps = max(int(max_steps), 0)
        self._eps = max(float(eps), 0.0)
        self._step = 0

        self._delta_sign_pos: np.ndarray | None = None
        self._delta_sign_neg: np.ndarray | None = None
        self._delta_like_counter = 0
        self._valid_counter = 0

    @override
    def on_episode_start(self) -> None:
        self._step = 0
        self._delta_sign_pos = None
        self._delta_sign_neg = None
        self._delta_like_counter = 0
        self._valid_counter = 0
        logger.info(
            "ACT_STATE_DIAG enabled (interval=%s, max_steps=%s, eps=%s)",
            self._interval,
            self._max_steps,
            self._eps,
        )

    @override
    def on_step(self, observation: dict, action: dict) -> None:
        self._step += 1
        if self._max_steps > 0 and self._step > self._max_steps:
            return

        state = self._extract_vector(observation, "observation.state")
        action_vec = self._extract_vector(action, "actions")
        if state is None or action_vec is None:
            if (self._step - 1) % self._interval == 0:
                logger.info("ACT_STATE_DIAG step=%s skipped (missing observation.state or actions)", self._step)
            return

        dim = min(state.shape[0], action_vec.shape[0])
        state = state[:dim]
        action_vec = action_vec[:dim]
        delta = action_vec - state

        if self._delta_sign_pos is None or self._delta_sign_pos.shape[0] != dim:
            self._delta_sign_pos = np.zeros(dim, dtype=np.int64)
            self._delta_sign_neg = np.zeros(dim, dtype=np.int64)
        assert self._delta_sign_neg is not None

        self._delta_sign_pos += (delta > self._eps).astype(np.int64)
        self._delta_sign_neg += (delta < -self._eps).astype(np.int64)

        joint_dims = [idx for idx in range(dim) if idx not in (6, 13)]
        if joint_dims:
            state_joint = state[joint_dims]
            action_joint = action_vec[joint_dims]
            delta_joint = delta[joint_dims]
        else:
            state_joint = state
            action_joint = action_vec
            delta_joint = delta

        state_mag = float(np.mean(np.abs(state_joint)))
        action_mag = float(np.mean(np.abs(action_joint)))
        delta_mag = float(np.mean(np.abs(delta_joint)))
        delta_to_state_ratio = delta_mag / (state_mag + 1e-6)
        action_to_state_ratio = action_mag / (state_mag + 1e-6)

        possible_delta_as_absolute = (delta_to_state_ratio > 0.7) and (action_to_state_ratio < 0.7)
        self._valid_counter += 1
        if possible_delta_as_absolute:
            self._delta_like_counter += 1

        sign_mask = (np.abs(state_joint) > self._eps) & (np.abs(action_joint) > self._eps)
        if np.any(sign_mask):
            sign_same_rate = float(np.mean(np.sign(state_joint[sign_mask]) == np.sign(action_joint[sign_mask])))
        else:
            sign_same_rate = float("nan")

        if (self._step - 1) % self._interval != 0:
            return

        logger.info("ACT_STATE_DIAG step=%s", self._step)
        logger.info("ACT_STATE_DIAG %s", _vec_line("state_raw", state))
        logger.info("ACT_STATE_DIAG %s", _vec_line("action_raw", action_vec))
        logger.info("ACT_STATE_DIAG %s", _vec_line("action_minus_state", delta))
        logger.info(
            "ACT_STATE_DIAG magnitudes: state=%.4f action=%.4f delta=%.4f delta/state=%.3f action/state=%.3f sign_same=%.3f",
            state_mag,
            action_mag,
            delta_mag,
            delta_to_state_ratio,
            action_to_state_ratio,
            sign_same_rate,
        )
        logger.info(
            "ACT_STATE_DIAG delta_like_votes=%s/%s heuristic=%s",
            self._delta_like_counter,
            self._valid_counter,
            "YES" if possible_delta_as_absolute else "NO",
        )

        pos_counts = self._delta_sign_pos.tolist()
        neg_counts = self._delta_sign_neg.tolist()
        sign_summary = [f"d{i}:{pos_counts[i]}/-{neg_counts[i]}" for i in range(dim)]
        logger.info("ACT_STATE_DIAG delta_sign_counts(+/-)=%s", ", ".join(sign_summary))

    @override
    def on_episode_end(self) -> None:
        if self._valid_counter <= 0:
            return
        ratio = self._delta_like_counter / self._valid_counter
        logger.info(
            "ACT_STATE_DIAG episode_summary delta_like_ratio=%.3f (%s/%s)",
            ratio,
            self._delta_like_counter,
            self._valid_counter,
        )

    def _extract_vector(self, data: dict[str, Any], key: str) -> np.ndarray | None:
        if key not in data:
            return None
        vec = np.asarray(data[key], dtype=np.float64)
        if vec.ndim == 2:
            vec = vec[0]
        return vec.reshape(-1)
