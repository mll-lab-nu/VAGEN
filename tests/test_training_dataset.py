from vagen.training.dataset import AgenticDataset


def test_agentic_dataset_honours_verl_max_samples(tmp_path):
    spec = tmp_path / "envs.yaml"
    spec.write_text(
        """envs:
  - name: Sokoban
    n_envs: 5
    seed: [1, 5, 1]
"""
    )

    dataset = AgenticDataset(
        data_files=str(spec),
        config={"base_seed": 0},
        max_samples=2,
    )

    assert len(dataset) == 2


def test_agentic_dataset_keeps_all_samples_without_a_limit(tmp_path):
    spec = tmp_path / "envs.yaml"
    spec.write_text(
        """envs:
  - name: Sokoban
    n_envs: 3
    seed: [1, 3, 1]
"""
    )

    assert len(AgenticDataset(data_files=str(spec), config={}, max_samples=-1)) == 3
