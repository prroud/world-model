import numpy as np

from scripts.validate_dataset import action_statistics, inspect_episode, scalar_statistics


def write_episode(path, actions=None, rewards=None, dones=None) -> None:
    actions = np.asarray(actions if actions is not None else [[-1.0, 0.5, 0.0], [1.0, 1.0, 0.2]], dtype=np.float32)
    rewards = np.asarray(rewards if rewards is not None else [1.0, 2.0], dtype=np.float32)
    dones = np.asarray(dones if dones is not None else [False, True], dtype=np.bool_)
    frames = np.zeros((2, 64, 64, 3), dtype=np.uint8)
    with path.open("wb") as archive:
        np.savez_compressed(
            archive,
            observations=frames,
            actions=actions,
            rewards=rewards,
            dones=dones,
            episode_index=np.asarray(0, dtype=np.int32),
            episode_seed=np.asarray(10, dtype=np.int64),
        )


def test_valid_episode_reports_aligned_data_and_statistics(tmp_path) -> None:
    episode_path = tmp_path / "episode_00000.npz"
    write_episode(episode_path)

    report = inspect_episode(episode_path)

    assert report["errors"] == []
    assert report["frames"] == 2
    assert report["episode_return"] == 3.0
    assert report["mean_step_reward"] == 1.5
    assert report["action_statistics"]["steering"]["mean"] == 0.0


def test_episode_reports_out_of_range_actions(tmp_path) -> None:
    episode_path = tmp_path / "episode_00000.npz"
    write_episode(episode_path, actions=[[2.0, 0.5, 0.0], [0.0, 0.4, 0.0]])

    report = inspect_episode(episode_path)

    assert "actions are outside the CarRacing action range" in report["errors"]


def test_statistics_reject_invalid_values() -> None:
    assert action_statistics(np.asarray([[np.nan, 0.5, 0.0]])) is None
    assert scalar_statistics(np.asarray([1.0, np.inf])) is None
    assert scalar_statistics(np.asarray([], dtype=np.float32)) is None
