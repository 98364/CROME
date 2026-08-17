from pathlib import Path

from crome_identification import config


def test_config_resolution_falls_back_to_environment_prefix(monkeypatch, tmp_path):
    prefix_configs = tmp_path / "configs"
    prefix_configs.mkdir()
    (prefix_configs / "installed.yaml").write_text("source: prefix\n", encoding="utf-8")

    monkeypatch.setattr(config, "SOURCE_CONFIG_DIR", tmp_path / "missing-source")
    monkeypatch.setattr(config, "INSTALLED_CONFIG_DIR", tmp_path / "missing-site-packages")
    monkeypatch.setattr(config, "PREFIX_CONFIG_DIR", prefix_configs, raising=False)

    assert config.resolve_config("installed") == {"source": "prefix"}
