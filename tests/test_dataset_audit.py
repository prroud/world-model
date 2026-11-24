import numpy as np

from scripts.validate_dataset import inspect_episode


def save_episode(path, *, actions=None, dones=None):
    frames = np.zeros((3, 64, 64, 3), dtype=np.uint8)
    if actions is None:
        actions = np.zeros((3, 3), dtype=np.float32)
    if dones is None:
        dones = np.asarray([False, False, True], dtype=np.bool_)
    np.savez_compressed(
        path,
        observations=frames,
        actions=actions,
        rewards=np.zeros(3, dtype=np.float32),
        dones=dones,
        episode_index=np.asarray(0, dtype=np.int32),
        episode_seed=np.asarray(123, dtype=np.int64),
    )


def test_audit_accepts_valid_episode_and_reports_blank_frames(tmp_path):
    path = tmp_path / "episode_00000.npz"
    save_episode(path)

    report = inspect_episode(path)

    assert report["errors"] == []
    assert report["frames"] == 3
    assert report["episode_seed"] == 123
    assert "3 fully black observation(s)" in report["warnings"]


def test_audit_rejects_invalid_action_range(tmp_path):
    path = tmp_path / "episode_00000.npz"
    actions = np.zeros((3, 3), dtype=np.float32)
    actions[1, 1] = 1.5
    save_episode(path, actions=actions)

    report = inspect_episode(path)

    assert "actions are outside the CarRacing action range" in report["errors"]


def test_audit_rejects_incorrect_episode_boundary(tmp_path):
    path = tmp_path / "episode_00000.npz"
    save_episode(path, dones=np.zeros(3, dtype=np.bool_))

    report = inspect_episode(path)

    assert "dones must be false except for a true final entry" in report["errors"]
