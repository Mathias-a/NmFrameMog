from __future__ import annotations

from dataclasses import dataclass


def normalize_json_value(raw: object) -> object:
    if raw is None or isinstance(raw, (str, int, float, bool)):
        return raw
    if isinstance(raw, list):
        return [normalize_json_value(item) for item in raw]
    if isinstance(raw, dict):
        normalized: dict[str, object] = {}
        for key, value in raw.items():
            if not isinstance(key, str):
                raise ValueError("JSON object keys must be strings.")
            normalized[key] = normalize_json_value(value)
        return normalized
    raise ValueError(f"Unsupported JSON value type: {type(raw).__name__}")


def ensure_json_object(raw: object) -> dict[str, object]:
    normalized = normalize_json_value(raw)
    if not isinstance(normalized, dict):
        raise ValueError("Expected a JSON object.")
    return normalized


@dataclass(frozen=True)
class AuthConfig:
    token: str
    header_name: str = "Authorization"
    scheme: str = "Bearer"

    def headers(self) -> dict[str, str]:
        if self.scheme:
            return {self.header_name: f"{self.scheme} {self.token}"}
        return {self.header_name: self.token}
