"""Configuration loading.

configs/config.yaml holds the defaults (local development, Colab). An
environment can override part of it with an overlay file, named by the
RAG_CONFIG_OVERLAY environment variable, that contains only what differs
(for Docker: backends and service URLs).
"""

import os

import yaml


def _deep_merge(base: dict, override: dict) -> dict:
    """Return base updated with override, merging nested sections."""
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _read_yaml(path: str) -> dict:
    with open(path, encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def load_config(config_path="configs/config.yaml") -> dict:
    """Load the base config and apply the overlay, if one is set."""
    config = _read_yaml(config_path)

    overlay_path = os.getenv("RAG_CONFIG_OVERLAY")
    if overlay_path:
        config = _deep_merge(config, _read_yaml(overlay_path))

    return config
