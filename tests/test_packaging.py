import os
from pathlib import Path
import subprocess
import sys


def test_wheel_cli_runs_outside_repository(tmp_path):
    repo = Path(__file__).resolve().parents[1]
    wheel_dir = tmp_path / "wheel"
    target = tmp_path / "installed"
    run_dir = tmp_path / "run"
    output = run_dir / "results" / "raw"
    wheel_dir.mkdir()
    run_dir.mkdir()

    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "wheel",
            str(repo),
            "--no-deps",
            "--no-build-isolation",
            "--wheel-dir",
            str(wheel_dir),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    wheel = next(wheel_dir.glob("*.whl"))
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            str(wheel),
            "--no-deps",
            "--target",
            str(target),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    env = os.environ.copy()
    env["PYTHONPATH"] = str(target)
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "from crome_identification.cli import main; raise SystemExit(main(['cs02']))",
        ],
        cwd=run_dir,
        env=env,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert (output / "cs02_smoke.json").exists()
