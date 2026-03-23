# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Register LeRobot ``PI0Config`` with draccus without importing ``lerobot.policies`` package ``__init__``.

``PreTrainedConfig.from_pretrained`` decodes ``\"type\": \"pi0\"`` only if ``@register_subclass(\"pi0\")``
has run. Importing ``lerobot.policies.pi0.configuration_pi0`` normally loads
``lerobot/policies/__init__.py``, which eagerly imports every policy and pulls in heavy deps
(e.g. PI0Policy → transformers). We load only ``configuration_rtc.py`` and ``configuration_pi0.py``
from disk into ``sys.modules`` after installing lightweight namespace stubs for
``lerobot.policies`` / ``lerobot.policies.rtc`` / ``lerobot.policies.pi0`` when those packages are
not already loaded."""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import sys
import tempfile
import types
from pathlib import Path
from typing import Any

import lerobot
from huggingface_hub import hf_hub_download
from huggingface_hub.constants import CONFIG_NAME
from huggingface_hub.errors import HfHubHTTPError
from lerobot.configs.policies import PreTrainedConfig

from ..utils import logging

logger = logging.get_logger(__name__)

_RTC_MOD = "lerobot.policies.rtc.configuration_rtc"
_PI0_MOD = "lerobot.policies.pi0.configuration_pi0"


def _lerobot_src_root() -> Path:
    return Path(lerobot.__file__).resolve().parent


def _inject_policy_package_stubs() -> None:
    root = _lerobot_src_root()
    policies_dir = root / "policies"
    if "lerobot.policies" not in sys.modules:
        pol = types.ModuleType("lerobot.policies")
        pol.__path__ = [str(policies_dir)]  # type: ignore[attr-defined]
        sys.modules["lerobot.policies"] = pol
    if "lerobot.policies.rtc" not in sys.modules:
        rtc = types.ModuleType("lerobot.policies.rtc")
        rtc.__path__ = [str(policies_dir / "rtc")]  # type: ignore[attr-defined]
        sys.modules["lerobot.policies.rtc"] = rtc
    if "lerobot.policies.pi0" not in sys.modules:
        pi0 = types.ModuleType("lerobot.policies.pi0")
        pi0.__path__ = [str(policies_dir / "pi0")]  # type: ignore[attr-defined]
        sys.modules["lerobot.policies.pi0"] = pi0


def _exec_module_from_file(qualified_name: str, path: Path) -> types.ModuleType:
    if qualified_name in sys.modules:
        return sys.modules[qualified_name]
    spec = importlib.util.spec_from_file_location(qualified_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {qualified_name} from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[qualified_name] = mod
    spec.loader.exec_module(mod)
    return mod


def _register_pi0_via_stub_bootstrap() -> bool:
    """Return True if PI0Config was loaded and ``pi0`` is registered."""
    root = _lerobot_src_root()
    rtc_path = root / "policies" / "rtc" / "configuration_rtc.py"
    pi0_path = root / "policies" / "pi0" / "configuration_pi0.py"
    if not rtc_path.is_file() or not pi0_path.is_file():
        return False
    _inject_policy_package_stubs()
    _exec_module_from_file(_RTC_MOD, rtc_path)
    _exec_module_from_file(_PI0_MOD, pi0_path)
    reg = getattr(PreTrainedConfig, "_choice_registry", {})
    return "pi0" in reg


def _register_pi0_via_full_import() -> bool:
    """Fallback: normal import (runs ``lerobot.policies`` ``__init__``)."""
    import importlib

    importlib.import_module(_PI0_MOD)
    reg = getattr(PreTrainedConfig, "_choice_registry", {})
    return "pi0" in reg


def ensure_pi0_config_registered_with_draccus() -> None:
    """Ensure draccus can decode ``type: pi0`` before ``PreTrainedConfig.from_pretrained``."""
    reg = getattr(PreTrainedConfig, "_choice_registry", {})
    if "pi0" in reg:
        return

    # If the real ``lerobot.policies`` package (``__init__.py``) is already loaded, only full import is safe.
    policies_mod = sys.modules.get("lerobot.policies")
    if policies_mod is not None and getattr(policies_mod, "__file__", None):
        if _register_pi0_via_full_import():
            return
        raise RuntimeError(
            f"Could not register PI0Config via {_PI0_MOD} after lerobot.policies was already imported."
        )

    try:
        if _register_pi0_via_stub_bootstrap():
            return
    except Exception as e:
        logger.warning_rank0(
            f"Stub bootstrap for PI0Config failed ({e!r}); trying full import of {_PI0_MOD}."
        )

    if _register_pi0_via_full_import():
        return

    raise RuntimeError(
        "Could not register LeRobot PI0Config with draccus (choice 'pi0' missing). "
        "Ensure lerobot is installed and policies sources exist under the lerobot package."
    )


_CACHED_PI0_TOP_LEVEL_KEYS: frozenset[str] | None = None


def _dataclass_field_names(cls: type) -> set[str]:
    names: set[str] = set()
    for base in cls.__mro__:
        fs = getattr(base, "__dataclass_fields__", None)
        if fs:
            names.update(fs.keys())
    return names


def _pi0_allowed_top_level_json_keys() -> frozenset[str]:
    """Field names accepted by draccus for ``PI0Config`` (plus ``type`` for the choice decoder)."""
    global _CACHED_PI0_TOP_LEVEL_KEYS
    if _CACHED_PI0_TOP_LEVEL_KEYS is not None:
        return _CACHED_PI0_TOP_LEVEL_KEYS
    ensure_pi0_config_registered_with_draccus()
    from lerobot.policies.pi0.configuration_pi0 import PI0Config

    names = _dataclass_field_names(PI0Config) | {"type"}
    _CACHED_PI0_TOP_LEVEL_KEYS = frozenset(names)
    return _CACHED_PI0_TOP_LEVEL_KEYS


# Lingbot-VLA modeling reads these attributes on ``config``; they are not LeRobot ``PI0Config`` fields.
# Checkpoint JSON often stores them under the same names. Map LeRobot field -> Lingbot alias after load.
#
# | LeRobot ``PI0Config`` (official) | Lingbot modeling (alias / extra) |
# |----------------------------------|----------------------------------|
# | ``num_inference_steps``          | also expose as ``num_steps`` (flow-matching ``dt``) |
#
# Keys with no PI0 equivalent stay only as dynamic attributes (setattr after parse).
def _normalize_lingbot_pi0_json_for_draccus(raw: dict[str, Any]) -> dict[str, Any]:
    """Drop or rename keys so draccus can decode ``PI0Config``; return extras for ``setattr`` on the live config."""
    allowed = _pi0_allowed_top_level_json_keys()
    extras: dict[str, Any] = {}

    # Old Lingbot name ``num_steps`` (flow horizon / denoising count) -> LeRobot ``num_inference_steps``.
    if "num_steps" in raw:
        if "num_inference_steps" not in raw:
            raw["num_inference_steps"] = raw.pop("num_steps")
        else:
            extras["num_steps"] = raw.pop("num_steps")

    for k in list(raw.keys()):
        if k not in allowed:
            extras[k] = raw.pop(k)

    return extras


def _apply_lingbot_pi0_runtime_aliases(cfg: PreTrainedConfig, extras: dict[str, Any]) -> None:
    """Attach Lingbot-only attributes expected by ``modeling_*`` (``num_steps``, ``proj_width``, …)."""
    for k, v in extras.items():
        try:
            setattr(cfg, k, v)
        except Exception:
            logger.warning_rank0(f"Could not set policy config attribute {k!r} on {type(cfg).__name__}")

    # Flow matching uses ``config.num_steps``; LeRobot only defines ``num_inference_steps``.
    if not hasattr(cfg, "num_steps"):
        nis = getattr(cfg, "num_inference_steps", None)
        if nis is not None:
            setattr(cfg, "num_steps", nis)


def pretrained_policy_config_from_pretrained(
    pretrained_name_or_path: str | Path,
    *,
    force_download: bool = False,
    resume_download: bool | None = None,
    proxies: dict[str, Any] | None = None,
    token: str | bool | None = None,
    cache_dir: str | Path | None = None,
    local_files_only: bool = False,
    revision: str | None = None,
    **policy_kwargs: Any,
) -> PreTrainedConfig:
    """Like ``PreTrainedConfig.from_pretrained``, but strips Lingbot-only JSON keys before draccus parse.

    Re-applies stripped keys on the loaded config instance (for code that reads them at runtime).
    """
    ensure_pi0_config_registered_with_draccus()

    model_id = str(pretrained_name_or_path)
    config_file: str | None = None
    if Path(model_id).is_dir():
        if CONFIG_NAME in os.listdir(model_id):
            config_file = os.path.join(model_id, CONFIG_NAME)
        else:
            raise FileNotFoundError(f"{CONFIG_NAME} not found in {Path(model_id).resolve()}")
    else:
        try:
            config_file = hf_hub_download(
                repo_id=model_id,
                filename=CONFIG_NAME,
                revision=revision,
                cache_dir=cache_dir,
                force_download=force_download,
                proxies=proxies,
                resume_download=resume_download,
                token=token,
                local_files_only=local_files_only,
            )
        except HfHubHTTPError as e:
            raise FileNotFoundError(f"{CONFIG_NAME} not found on the HuggingFace Hub in {model_id}") from e

    with open(config_file) as f:
        raw = json.load(f)

    extras = _normalize_lingbot_pi0_json_for_draccus(raw)

    td = tempfile.mkdtemp(prefix="lingbot_lerobot_policy_cfg_")
    try:
        cleaned = Path(td) / CONFIG_NAME
        with open(cleaned, "w") as f:
            json.dump(raw, f, indent=4)
        cfg = PreTrainedConfig.from_pretrained(
            td,
            force_download=force_download,
            resume_download=resume_download,
            proxies=proxies,
            token=token,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
            revision=revision,
            **policy_kwargs,
        )
    finally:
        shutil.rmtree(td, ignore_errors=True)

    _apply_lingbot_pi0_runtime_aliases(cfg, extras)
    return cfg
