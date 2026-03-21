from __future__ import annotations

import pytest  # pyright: ignore[reportMissingImports]

from src.ng_data.eval.splits import SplitConfigValidationError, validate_split_manifest


def test_validate_split_manifest_rejects_image_leakage_across_sets() -> None:
    with pytest.raises(
        SplitConfigValidationError,
        match="train set contains holdout image_ids",
    ):
        validate_split_manifest(
            {
                "holdout": {"image_ids": [3]},
                "folds": [
                    {
                        "train_image_ids": [1, 3],
                        "val_image_ids": [2],
                    },
                    {
                        "train_image_ids": [1, 2],
                        "val_image_ids": [],
                    },
                ],
            }
        )


def test_generated_splits_assign_each_image_id_once_to_holdout_or_validation() -> None:
    payload: dict[str, object] = {
        "holdout": {"image_ids": [10]},
        "folds": [
            {"train_image_ids": [12, 13], "val_image_ids": [11]},
            {"train_image_ids": [11, 13], "val_image_ids": [12]},
            {"train_image_ids": [11, 12], "val_image_ids": [13]},
            {"train_image_ids": [11, 12, 13], "val_image_ids": []},
        ],
    }

    validate_split_manifest(payload)

    holdout = payload["holdout"]
    folds = payload["folds"]
    assert isinstance(holdout, dict)
    assert isinstance(folds, list)

    holdout_ids = holdout["image_ids"]
    assert isinstance(holdout_ids, list)

    assigned_once = holdout_ids + [
        image_id
        for fold in folds
        for image_id in (fold["val_image_ids"] if isinstance(fold, dict) else [])
    ]
    assert sorted(assigned_once) == [10, 11, 12, 13]
    assert len(assigned_once) == len(set(assigned_once))
