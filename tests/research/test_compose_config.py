"""Static and daemon-free checks for the containerized research runtime."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[2]
COMPOSE_PATH = ROOT / "docker-compose.yml"
DOCKERFILE_PATH = ROOT / "Dockerfile"


@pytest.fixture(scope="module")
def compose_config() -> dict[str, object]:
    parsed = yaml.safe_load(COMPOSE_PATH.read_text(encoding="utf-8"))
    assert isinstance(parsed, dict)
    return parsed


def _volume_targets(service: dict[str, object]) -> dict[str, tuple[str, bool]]:
    result: dict[str, tuple[str, bool]] = {}
    for volume in service.get("volumes", []):
        if isinstance(volume, str):
            source, target, *options = volume.split(":")
            result[target] = (source, "ro" in options)
        else:
            result[str(volume["target"])] = (
                str(volume["source"]),
                bool(volume.get("read_only", False)),
            )
    return result


def _environment(service: dict[str, object]) -> dict[str, str]:
    environment = service.get("environment", {})
    if isinstance(environment, dict):
        return {str(key): str(value) for key, value in environment.items()}
    result = {}
    for item in environment:
        key, _, value = str(item).partition("=")
        result[key] = value
    return result


def test_compose_has_required_services(compose_config):
    assert {
        "research-bot",
        "research-runner",
        "ollama",
        "ollama-init",
        "test",
    } <= set(compose_config["services"])


def test_research_services_use_fixed_non_root_identity(compose_config):
    services = compose_config["services"]
    for name in ("research-bot", "research-runner", "test"):
        assert services[name]["user"] == "10001:10001"


def test_research_state_mounts_match_config_derived_paths(compose_config):
    service = compose_config["services"]["research-bot"]
    environment = _environment(service)
    assert environment["RESEARCH_DATA_DIR"] == "/app/data"
    mounts = _volume_targets(service)
    assert mounts["/app/data"] == ("research-data", False)
    assert mounts["/app/data/reports"] == ("research-reports", False)
    assert mounts["/app/data/cache"] == ("research-cache", False)
    assert mounts["/app/data/source_cache"] == ("research-cache", False)
    assert mounts["/app/data/backups"] == ("research-backups", False)
    assert mounts["/app/backups"] == ("research-backups", False)
    assert mounts["/app/logs"] == ("research-logs", False)
    assert mounts["/app/audio_output"] == ("research-audio", False)


def test_named_volumes_cover_all_durable_runtime_data(compose_config):
    assert {
        "research-data",
        "research-reports",
        "research-backups",
        "research-cache",
        "research-logs",
        "research-audio",
        "ollama-models",
    } <= set(compose_config["volumes"])


def test_research_services_mount_secret_directory_read_only(compose_config):
    for name in ("research-bot", "research-runner"):
        mounts = _volume_targets(compose_config["services"][name])
        assert mounts["/run/secrets"] == ("./secrets", True)


def test_scheduler_and_runner_have_separate_safe_entrypoints(compose_config):
    services = compose_config["services"]
    assert services["research-bot"]["command"] == [
        "python",
        "-m",
        "news_bot.research.scheduler",
    ]
    assert services["research-runner"]["profiles"] == ["runner"]
    assert services["research-runner"]["entrypoint"] == [
        "python",
        "-m",
        "news_bot.research.cli",
    ]
    assert services["research-runner"]["command"] == ["--help"]
    assert services["test"]["profiles"] == ["test"]
    assert services["test"]["build"]["target"] == "development"


def test_healthcheck_is_local_and_migrates_then_checks_sqlite(compose_config):
    health = compose_config["services"]["research-bot"]["healthcheck"]
    command = " ".join(health["test"])
    assert "include_flex=False" in command
    assert "include_model_secret=False" in command
    assert ".migrate()" in command
    assert "quick_check" in command
    lowered = command.lower()
    assert "openai" not in lowered
    assert "flexwebservice" not in lowered
    assert "ollama" not in lowered


def test_ollama_has_healthcheck_model_volume_and_init_dependency(compose_config):
    services = compose_config["services"]
    ollama = services["ollama"]
    assert "healthcheck" in ollama
    assert _volume_targets(ollama)["/root/.ollama"] == (
        "ollama-models",
        False,
    )
    assert services["ollama-init"]["depends_on"]["ollama"]["condition"] == (
        "service_healthy"
    )


def test_ollama_init_validates_model_waits_and_is_idempotent():
    script = (ROOT / "scripts" / "ollama-init.sh").read_text(encoding="utf-8")
    assert "OLLAMA_RESEARCH_MODEL" in script
    assert "ollama list" in script
    assert "ollama pull" in script
    assert "grep -E" in script
    assert script.index("ollama list") < script.index("ollama pull")
    assert "sleep" in script


@pytest.mark.parametrize(
    "unsafe_model",
    ("-leading", "trailing/", "double//slash", "two:tags:bad", "../escape"),
)
def test_ollama_init_rejects_unsafe_models_before_cli_use(
    tmp_path: Path, unsafe_model: str
):
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    log = tmp_path / "ollama.log"
    ollama = fake_bin / "ollama"
    ollama.write_text(f"#!/bin/sh\necho called >> {log}\n", encoding="utf-8")
    ollama.chmod(0o755)
    environment = os.environ.copy()
    environment.update(
        OLLAMA_RESEARCH_MODEL=unsafe_model,
        PATH=f"{fake_bin}:{environment['PATH']}",
    )
    result = subprocess.run(
        ["sh", str(ROOT / "scripts" / "ollama-init.sh")],
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 2
    assert not log.exists()


def test_ollama_init_does_not_pull_an_installed_model(tmp_path: Path):
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    log = tmp_path / "ollama.log"
    ollama = fake_bin / "ollama"
    ollama.write_text(
        "#!/bin/sh\n"
        f"printf '%s\\n' \"$*\" >> {log}\n"
        "if [ \"$1\" = list ]; then\n"
        "  printf 'NAME ID SIZE MODIFIED\\nllama3.1:8b abc 1GB now\\n'\n"
        "fi\n",
        encoding="utf-8",
    )
    ollama.chmod(0o755)
    environment = os.environ.copy()
    environment.update(
        OLLAMA_RESEARCH_MODEL="llama3.1:8b",
        PATH=f"{fake_bin}:{environment['PATH']}",
    )
    result = subprocess.run(
        ["sh", str(ROOT / "scripts" / "ollama-init.sh")],
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "pull" not in log.read_text(encoding="utf-8")


def test_healthcheck_command_runs_without_credentials_or_network(
    compose_config, tmp_path: Path
):
    health = compose_config["services"]["research-bot"]["healthcheck"]
    command = health["test"][-1]
    environment = os.environ.copy()
    for name in tuple(environment):
        if name.startswith("IBKR_FLEX_") or name.startswith("OPENAI_API_KEY"):
            environment.pop(name)
    environment.update(RESEARCH_ENABLED="true", RESEARCH_DATA_DIR=str(tmp_path))
    result = subprocess.run(
        [sys.executable, "-c", command],
        cwd=ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "research.db").is_file()


def test_backup_uses_online_api_checks_integrity_and_bounds_retention():
    script = (ROOT / "scripts" / "backup-research-db.sh").read_text(
        encoding="utf-8"
    )
    assert "source.backup(destination)" in script
    assert "PRAGMA integrity_check" in script
    assert 'Path("/app/backups")' in script
    assert "is_symlink()" in script
    assert "backups[14:]" in script
    assert "shutil.copy" not in script
    assert "cp " not in script


def test_image_has_weasyprint_libraries_assets_and_fixed_user():
    dockerfile = DOCKERFILE_PATH.read_text(encoding="utf-8")
    for library in (
        "libcairo2",
        "libpango-1.0-0",
        "libpangocairo-1.0-0",
        "libgdk-pixbuf-2.0-0",
        "libffi-dev",
        "shared-mime-info",
    ):
        assert library in dockerfile
    assert "--uid 10001" in dockerfile
    assert "--gid 10001" in dockerfile
    for path in (
        "/app/data",
        "/app/reports",
        "/app/cache",
        "/app/backups",
        "/app/logs",
        "/app/audio_output",
    ):
        assert path in dockerfile
    assert "COPY --chown=appuser:appuser news_bot/ ./news_bot/" in dockerfile
    assert "COPY --chown=appuser:appuser scripts/ ./scripts/" in dockerfile
    assert dockerfile.count("USER appuser") >= 2


def test_secret_material_and_runtime_state_are_excluded_from_context_and_git():
    dockerignore = (ROOT / ".dockerignore").read_text(encoding="utf-8")
    gitignore = (ROOT / ".gitignore").read_text(encoding="utf-8")
    env_example = (ROOT / "env.example").read_text(encoding="utf-8")
    assert "secrets/" in dockerignore
    assert "secrets/" in gitignore
    assert "IBKR_FLEX_TOKEN_FILE=/run/secrets/ibkr_flex_token" in env_example
    assert "IBKR_FLEX_QUERY_ID_FILE=/run/secrets/ibkr_flex_query_id" in env_example
    assert "OPENAI_API_KEY_FILE=/run/secrets/openai_api_key" in env_example
    assert "IBKR_FLEX_TOKEN=" not in env_example
    assert "OPENAI_API_KEY=" not in env_example


def test_compose_renders_when_docker_compose_is_available():
    if shutil.which("docker") is None:
        pytest.skip("Docker CLI is not installed")
    environment = os.environ.copy()
    environment.setdefault("COMPOSE_PROJECT_NAME", "newsletter-static-test")
    result = subprocess.run(
        ["docker", "compose", "config", "--quiet"],
        cwd=ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
