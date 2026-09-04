"""
Tests for shared utility functions.
"""

import json

import pandas as pd
import pytest

from src.utils.io import (
    ensure_directory,
    load_dataframe,
    load_json,
    save_dataframe,
    save_json,
)
from src.utils.reproducibility import (
    create_random_generator,
    set_random_seed,
)


def test_ensure_directory_creates_directory(tmp_path) -> None:
    """ensure_directory should create a missing directory."""
    directory = tmp_path / "new_directory"

    result = ensure_directory(directory)

    assert directory.exists()
    assert directory.is_dir()
    assert result == directory


def test_save_and_load_dataframe(tmp_path) -> None:
    """A saved CSV should be loaded back correctly."""
    dataframe = pd.DataFrame(
        {
            "requests": [100, 120, 150],
            "latency_ms": [50.0, 55.0, 60.0],
        }
    )

    file_path = tmp_path / "data" / "test_data.csv"

    saved_path = save_dataframe(
        dataframe=dataframe,
        file_path=file_path,
    )

    loaded_dataframe = load_dataframe(saved_path)

    assert saved_path.exists()

    pd.testing.assert_frame_equal(
        dataframe,
        loaded_dataframe,
    )


def test_save_dataframe_rejects_non_csv_file(tmp_path) -> None:
    """save_dataframe should reject unsupported file extensions."""
    dataframe = pd.DataFrame(
        {
            "value": [1, 2, 3],
        }
    )

    with pytest.raises(ValueError):
        save_dataframe(
            dataframe=dataframe,
            file_path=tmp_path / "test_data.json",
        )


def test_load_dataframe_raises_for_missing_file(tmp_path) -> None:
    """load_dataframe should raise an error for a missing file."""
    missing_file = tmp_path / "missing.csv"

    with pytest.raises(FileNotFoundError):
        load_dataframe(missing_file)


def test_save_and_load_json(tmp_path) -> None:
    """A saved JSON dictionary should be loaded back correctly."""
    data = {
        "data_type": "synthetic",
        "random_seed": 42,
        "baseline_requests": 100.0,
    }

    file_path = tmp_path / "metadata" / "test_metadata.json"

    saved_path = save_json(
        data=data,
        file_path=file_path,
    )

    loaded_data = load_json(saved_path)

    assert saved_path.exists()
    assert loaded_data == data


def test_saved_json_is_valid_json(tmp_path) -> None:
    """save_json should produce a valid JSON file."""
    data = {
        "scenario": "normal",
        "random_seed": 42,
    }

    file_path = tmp_path / "test.json"

    save_json(
        data=data,
        file_path=file_path,
    )

    with file_path.open(
        mode="r",
        encoding="utf-8",
    ) as file:
        loaded_data = json.load(file)

    assert loaded_data == data


def test_save_json_rejects_non_json_file(tmp_path) -> None:
    """save_json should reject unsupported file extensions."""
    with pytest.raises(ValueError):
        save_json(
            data={"value": 1},
            file_path=tmp_path / "test.csv",
        )


def test_load_json_raises_for_missing_file(tmp_path) -> None:
    """load_json should raise an error for a missing file."""
    missing_file = tmp_path / "missing.json"

    with pytest.raises(FileNotFoundError):
        load_json(missing_file)


def test_random_generator_is_reproducible() -> None:
    """The same seed should produce the same random sequence."""
    rng_one = create_random_generator(42)
    rng_two = create_random_generator(42)

    values_one = rng_one.random(10)
    values_two = rng_two.random(10)

    assert (values_one == values_two).all()


def test_random_generator_changes_with_different_seed() -> None:
    """Different seeds should normally produce different sequences."""
    rng_one = create_random_generator(42)
    rng_two = create_random_generator(123)

    values_one = rng_one.random(10)
    values_two = rng_two.random(10)

    assert not (values_one == values_two).all()


def test_negative_seed_raises_value_error() -> None:
    """Negative seeds should not be accepted."""
    with pytest.raises(ValueError):
        create_random_generator(-1)

    with pytest.raises(ValueError):
        set_random_seed(-1)