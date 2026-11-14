"""Load and validate config.yaml (the single place where assumptions live)."""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

REQUIRED_SECTIONS = [
    "well", "curves", "qc", "vsh", "toc", "porosity", "sw",
    "facies", "zonation", "cutoffs", "flags", "llm",
]


class ConfigError(ValueError):
    """Raised when config.yaml is missing something or holds an impossible value."""


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _num_or_auto(value: Any, name: str) -> None:
    if value == "auto" or _is_number(value):
        return
    raise ConfigError(f"{name} must be a number or 'auto', got {value!r}")


def validate_config(cfg: dict) -> None:
    for section in REQUIRED_SECTIONS:
        if section not in cfg or cfg[section] is None:
            raise ConfigError(f"config is missing the '{section}' section")

    por, sw, vsh, toc = cfg["porosity"], cfg["sw"], cfg["vsh"], cfg["toc"]

    if not (_is_number(por["rho_matrix"]) and _is_number(por["rho_fluid"])
            and por["rho_matrix"] > por["rho_fluid"] > 0):
        raise ConfigError("porosity: need rho_matrix > rho_fluid > 0 (g/cc)")
    if por["fluid_type"] not in ("liquid", "gas"):
        raise ConfigError("porosity.fluid_type must be 'liquid' or 'gas'")
    if por["phit_method"] not in ("density", "density_neutron"):
        raise ConfigError("porosity.phit_method must be 'density' or 'density_neutron'")
    if por.get("phie_method") not in ("density", "density_neutron"):
        raise ConfigError("porosity.phie_method must be 'density' or 'density_neutron'")
    for key in ("rho_shale", "nphi_shale"):
        _num_or_auto(por[key], f"porosity.{key}")

    for key in ("a", "m", "n", "rw"):
        if key not in sw or not _is_number(sw[key]) or sw[key] <= 0:
            raise ConfigError(f"sw.{key} must be a positive number, got {sw.get(key)!r}")
    _num_or_auto(sw["rsh"], "sw.rsh")
    for key in ("archie_porosity", "simandoux_porosity"):
        if sw[key] not in ("PHIT", "PHIE"):
            raise ConfigError(f"sw.{key} must be PHIT or PHIE")

    for key in ("gr_clean", "gr_shale"):
        _num_or_auto(vsh[key], f"vsh.{key}")
    if vsh["larionov_variant"] not in ("older", "tertiary"):
        raise ConfigError("vsh.larionov_variant must be 'older' or 'tertiary'")
    if vsh["use_for_downstream"] not in ("linear", "larionov"):
        raise ConfigError("vsh.use_for_downstream must be 'linear' or 'larionov'")

    if toc["method"] not in ("sonic", "density", "neutron"):
        raise ConfigError("toc.method must be sonic, density or neutron")
    for key in ("rt_baseline", "dt_baseline", "rhob_baseline", "nphi_baseline"):
        _num_or_auto(toc[key], f"toc.{key}")
    if not (_is_number(toc["lom"]) and 0 < toc["lom"] < 20):
        raise ConfigError("toc.lom looks wrong (expected roughly 6-12)")

    if cfg["qc"]["spike"]["window"] % 2 == 0:
        raise ConfigError("qc.spike.window must be odd")
    if cfg["qc"]["badhole_action"] not in ("null_density_neutron", "flag_only"):
        raise ConfigError("qc.badhole_action must be null_density_neutron or flag_only")
    if cfg["qc"]["spike"]["action"] not in ("set_nan", "flag_only"):
        raise ConfigError("qc.spike.action must be set_nan or flag_only")
    if cfg["zonation"]["method"] not in ("auto", "tops"):
        raise ConfigError("zonation.method must be 'auto' or 'tops'")


def load_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    if not isinstance(cfg, dict):
        raise ConfigError(f"{path} did not parse to a mapping")
    validate_config(cfg)
    return cfg


def clone(obj: dict) -> dict:
    return copy.deepcopy(obj)
