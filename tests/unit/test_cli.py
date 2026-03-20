"""Tests for the EvalPulse CLI."""

import subprocess
import sys


class TestCLI:
    """Test CLI commands."""

    def test_help_output(self):
        """CLI should show help text."""
        result = subprocess.run(
            [sys.executable, "-m", "evalpulse.cli", "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "EvalPulse" in result.stdout

    def test_init_creates_config(self, tmp_path, monkeypatch):
        """evalpulse init should create evalpulse.yml."""
        monkeypatch.chdir(tmp_path)
        result = subprocess.run(
            [sys.executable, "-m", "evalpulse.cli", "init"],
            capture_output=True,
            text=True,
            cwd=str(tmp_path),
        )
        assert result.returncode == 0
        assert (tmp_path / "evalpulse.yml").exists()

    def test_regression_missing_dataset(self):
        """Regression with missing dataset should fail."""
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "evalpulse.cli",
                "regression",
                "run",
                "--dataset",
                "/nonexistent.json",
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode != 0
