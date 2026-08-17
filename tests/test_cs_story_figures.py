import hashlib
import json
from pathlib import Path

from PIL import Image

from figures.cs_paper_figures import build_story_figures


def test_story_figure_package_has_one_schematic_and_three_rq_figures(tmp_path):
    exported = build_story_figures(output_dir=tmp_path)

    assert set(exported) == {"pipeline", "rq1", "rq2", "rq3"}
    for paths in exported.values():
        assert {path.suffix for path in paths} == {".pdf", ".png"}
        png = next(path for path in paths if path.suffix == ".png")
        with Image.open(png) as image:
            assert image.width >= 800
            assert image.height >= 300

    assert (tmp_path / "manifest.json").exists()
    assert (tmp_path / "source_data.csv").exists()
    assert (tmp_path / "figure_00_evidence_certificate_route.pdf").exists()
    assert (tmp_path / "figure_03_routing_ablation.pdf").exists()

    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    revision_summary = (
        Path(__file__).resolve().parents[1]
        / "results"
        / "revision_20260812"
        / "summaries"
        / "cs03_main.json"
    )
    assert manifest["source_summaries"]["cs03_main.json"] == hashlib.sha256(
        revision_summary.read_bytes()
    ).hexdigest()
