from __future__ import annotations

from src.ng_data.data.config import DataConfig, DataConfigValidationError


def validate_data_config(config: DataConfig) -> None:
    if config.schema_version != 1:
        raise DataConfigValidationError(
            f"Unsupported schema version: {config.schema_version}"
        )

    processed = config.processed_layout
    if processed.manifest_path == processed.annotations_path:
        raise DataConfigValidationError(
            "manifest_path must differ from annotations_path"
        )
    if processed.categories_path == processed.product_index_path:
        raise DataConfigValidationError(
            "categories_path must differ from product_index_path"
        )
