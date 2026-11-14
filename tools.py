"""Tools the LLM can call during chat. Each tool RE-RUNS the Python petrophysics on the
depth samples of a zone with the changed parameter(s) and returns the base case, the
what-if case and the difference - so the LLM never has to subtract, average or convert.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pandas as pd

from .petrophysics import compute_all
from .zonation import petro_averages

SW_KEYS = ["sw_archie_mean", "sw_simandoux_mean", "sw_archie_porosity_weighted",
           "sw_simandoux_porosity_weighted", "sw_archie_above_one_fraction"]
POR_KEYS = ["phit_mean", "phie_mean", "low_porosity_fraction", "rwa_median"]
VSH_KEYS = ["vsh_linear_mean", "vsh_larionov_mean"]
TOC_KEYS = ["toc_mean_wtpct", "toc_max_wtpct", "net_toc_thickness"]
DECIMALS = {"toc_mean_wtpct": 2, "toc_max_wtpct": 2, "net_toc_thickness": 2, "rwa_median": 4}
RANK_METRICS = ["toc_mean_wtpct", "toc_max_wtpct", "net_toc_thickness", "phit_mean", "phie_mean",
                "sw_archie_mean", "sw_simandoux_mean", "vsh_larionov_mean", "vsh_linear_mean"]

_ZONE = {"type": "string", "description": "Zone name such as 'Z3' (or just '3'), or 'all' for every zone."}


def _num(desc: str) -> dict:
    return {"type": "number", "description": desc}


TOOLS = [
    {"name": "list_zones",
     "description": "List all zones with top, base, thickness, dominant facies and the low-confidence flag.",
     "input_schema": {"type": "object", "properties": {}, "required": []}},
    {"name": "get_zone",
     "description": "Return the full stored summary (averages, facies, QC flags) of one zone.",
     "input_schema": {"type": "object", "properties": {"zone": _ZONE}, "required": ["zone"]}},
    {"name": "compute_sw",
     "description": ("Re-run water saturation (Archie and Simandoux) in Python for a zone with different "
                     "parameters. Use for any 'what if Rw / m / n / a / Rsh is ...' question. "
                     "Omitted parameters keep their base value."),
     "input_schema": {"type": "object", "properties": {
         "zone": _ZONE,
         "rw": _num("Formation water resistivity at formation temperature, ohm.m."),
         "m": _num("Cementation exponent."),
         "n": _num("Saturation exponent."),
         "a": _num("Tortuosity factor."),
         "rsh": _num("Shale resistivity for Simandoux, ohm.m.")}, "required": ["zone"]}},
    {"name": "compute_vsh",
     "description": ("Re-run shale volume in Python for a zone with different gamma-ray end points or method. "
                     "Porosity (clay correction) and Sw are re-run too because they depend on Vsh."),
     "input_schema": {"type": "object", "properties": {
         "zone": _ZONE,
         "gr_clean": _num("Gamma ray of clean rock, API."),
         "gr_shale": _num("Gamma ray of shale, API."),
         "method": {"type": "string", "enum": ["linear", "larionov_older", "larionov_tertiary"],
                    "description": "Vsh model used downstream."}}, "required": ["zone"]}},
    {"name": "compute_porosity",
     "description": ("Re-run porosity (density / neutron, kerogen- and clay-corrected) in Python for a zone "
                     "with different matrix, fluid or shale-point values. Sw is re-run too."),
     "input_schema": {"type": "object", "properties": {
         "zone": _ZONE,
         "rho_matrix": _num("Matrix (grain) density, g/cc."),
         "rho_fluid": _num("Fluid density, g/cc."),
         "rho_shale": _num("Density of non-organic shale, g/cc."),
         "nphi_shale": _num("Neutron reading of non-organic shale, v/v."),
         "apply_kerogen_correction": {"type": "boolean", "description": "Switch the kerogen correction on or off."}},
         "required": ["zone"]}},
    {"name": "compute_toc",
     "description": ("Re-run Passey delta-log-R TOC in Python for a zone with a different LOM or baselines. "
                     "Porosity (kerogen correction) and Sw are re-run too."),
     "input_schema": {"type": "object", "properties": {
         "zone": _ZONE,
         "lom": _num("Level of organic metamorphism."),
         "rt_baseline": _num("Resistivity baseline in non-source shale, ohm.m."),
         "dt_baseline": _num("Sonic baseline in non-source shale, us/ft (sonic method)."),
         "rhob_baseline": _num("Density baseline, g/cc (density method)."),
         "nphi_baseline": _num("Neutron baseline, v/v (neutron method)."),
         "method": {"type": "string", "enum": ["sonic", "density", "neutron"]}}, "required": ["zone"]}},
    {"name": "rank_zones",
     "description": "Sort the zones by one stored average (Python does the sorting).",
     "input_schema": {"type": "object", "properties": {
         "metric": {"type": "string", "enum": RANK_METRICS},
         "order": {"type": "string", "enum": ["high_to_low", "low_to_high"]}}, "required": ["metric"]}},
]


class ToolError(ValueError):
    pass


class Session:
    """Holds the pipeline outputs and runs the tools."""

    def __init__(self, summary: dict, curves: pd.DataFrame):
        self.summary = summary
        self.curves = curves
        self.params = {k: copy.deepcopy(summary["parameters"][k]) for k in ("vsh", "toc", "porosity", "sw")}
        self.step = float(summary["well"]["step"])
        self.toc_cutoff = float(summary["parameters"]["cutoffs"]["toc_min_wtpct"])
        self.zones = {z["name"].upper(): z for z in summary["zones"]}
        self.by_id = {str(z["zone_id"]): z for z in summary["zones"]}
        self.calls: list[dict] = []          # every tool call, kept for verification

    @classmethod
    def load(cls, out_dir: str | Path) -> "Session":
        out_dir = Path(out_dir)
        summary = json.loads((out_dir / "zone_summary.json").read_text(encoding="utf-8"))
        curves = pd.read_csv(out_dir / "curves.csv")
        return cls(summary, curves)

    # ------------------------------------------------------------------ helpers
    def _zone(self, zone) -> dict:
        key = str(zone).strip().upper()
        if key in self.zones:
            return self.zones[key]
        if key in self.by_id:
            return self.by_id[key]
        if key.startswith("ZONE") and key[4:].strip() in self.by_id:
            return self.by_id[key[4:].strip()]
        raise ToolError(f"unknown zone {zone!r}; valid zones: {', '.join(z['name'] for z in self.summary['zones'])}")

    def _targets(self, zone) -> list[dict]:
        if str(zone).strip().lower() == "all":
            return list(self.summary["zones"])
        return [self._zone(zone)]

    def _rows(self, z: dict) -> pd.DataFrame:
        return self.curves[self.curves["ZONE"] == z["name"]]

    @staticmethod
    def _positive(name: str, value) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not value > 0:
            raise ToolError(f"{name} must be a positive number, got {value!r}")
        return float(value)

    def _what_if(self, zone, changes: dict, keys: list[str]) -> dict:
        """changes = {'sw.rw': 0.05, ...}. Runs the full chain so dependent results stay consistent."""
        params = copy.deepcopy(self.params)
        changed = {}
        for dotted, value in changes.items():
            section, *path = dotted.split(".")
            node = params[section]
            for part in path[:-1]:
                node = node[part]
            changed[dotted] = {"base": node[path[-1]], "what_if": value}
            node[path[-1]] = value
        if not params["vsh"]["gr_shale"] > params["vsh"]["gr_clean"]:
            raise ToolError("gr_shale must be larger than gr_clean")
        if not params["porosity"]["rho_matrix"] > params["porosity"]["rho_fluid"]:
            raise ToolError("rho_matrix must be larger than rho_fluid")

        results = []
        for z in self._targets(zone):
            sub = self._rows(z)
            if sub.empty:
                raise ToolError(f"no depth samples stored for zone {z['name']}")
            new_curves = compute_all(sub, params)
            new = petro_averages(new_curves, params, self.step, self.toc_cutoff)
            base = petro_averages(sub, self.params, self.step, self.toc_cutoff)
            diff = {}
            for k in keys:
                if new[k] is not None and base[k] is not None:
                    diff[k] = round(new[k] - base[k], DECIMALS.get(k, 3))
                else:
                    diff[k] = None
            results.append({
                "zone": z["name"], "top": z["top"], "base": z["base"], "n_samples": int(len(sub)),
                "low_confidence_zone": z["low_confidence"],
                "base_case": {k: base[k] for k in keys},
                "what_if": {k: new[k] for k in keys},
                "change_what_if_minus_base": diff,
            })
        return {
            "changed_parameters": changed if changed else "none (base case returned)",
            "results": results,
            "units": "Vsh, porosity, Sw and fractions in v/v; TOC in wt%; thickness in the LAS depth unit; Rwa in ohm.m",
            "computed_by": "Python: shale_assistant.petrophysics.compute_all re-run on the zone's depth samples",
            "note": ("Auto-picked values that are not listed under changed_parameters (shale point, Rsh, baselines) "
                     "were kept at their base values."),
        }

    # ------------------------------------------------------------------ tools
    def list_zones(self) -> dict:
        return {"depth_unit": self.summary["well"]["depth_unit"], "zones": [
            {"zone": z["name"], "top": z["top"], "base": z["base"], "thickness": z["thickness"],
             "dominant_facies": (z["facies"] or {}).get("dominant"),
             "low_confidence": z["low_confidence"]} for z in self.summary["zones"]]}

    def get_zone(self, zone) -> dict:
        return self._zone(zone)

    def compute_sw(self, zone, rw=None, m=None, n=None, a=None, rsh=None) -> dict:
        changes = {}
        for key, value in (("rw", rw), ("m", m), ("n", n), ("a", a), ("rsh", rsh)):
            if value is not None:
                changes[f"sw.{key}"] = self._positive(key, value)
        return self._what_if(zone, changes, SW_KEYS)

    def compute_vsh(self, zone, gr_clean=None, gr_shale=None, method=None) -> dict:
        changes = {}
        if gr_clean is not None:
            changes["vsh.gr_clean"] = float(gr_clean)
        if gr_shale is not None:
            changes["vsh.gr_shale"] = float(gr_shale)
        if method is not None:
            if method == "linear":
                changes["vsh.use_for_downstream"] = "linear"
            elif method in ("larionov_older", "larionov_tertiary"):
                changes["vsh.use_for_downstream"] = "larionov"
                changes["vsh.larionov_variant"] = method.split("_")[1]
            else:
                raise ToolError("method must be linear, larionov_older or larionov_tertiary")
        return self._what_if(zone, changes, VSH_KEYS + ["phie_mean"] + SW_KEYS)

    def compute_porosity(self, zone, rho_matrix=None, rho_fluid=None, rho_shale=None, nphi_shale=None,
                         apply_kerogen_correction=None) -> dict:
        changes = {}
        for key, value in (("rho_matrix", rho_matrix), ("rho_fluid", rho_fluid), ("rho_shale", rho_shale)):
            if value is not None:
                changes[f"porosity.{key}"] = self._positive(key, value)
        if nphi_shale is not None:
            changes["porosity.nphi_shale"] = float(nphi_shale)
        if apply_kerogen_correction is not None:
            changes["porosity.kerogen.apply"] = bool(apply_kerogen_correction)
        return self._what_if(zone, changes, POR_KEYS + SW_KEYS)

    def compute_toc(self, zone, lom=None, rt_baseline=None, dt_baseline=None, rhob_baseline=None,
                    nphi_baseline=None, method=None) -> dict:
        changes = {}
        if lom is not None:
            if not 0 < float(lom) < 20:
                raise ToolError("lom must be between 0 and 20")
            changes["toc.lom"] = float(lom)
        for key, value in (("rt_baseline", rt_baseline), ("dt_baseline", dt_baseline),
                           ("rhob_baseline", rhob_baseline)):
            if value is not None:
                changes[f"toc.{key}"] = self._positive(key, value)
        if nphi_baseline is not None:
            changes["toc.nphi_baseline"] = float(nphi_baseline)
        if method is not None:
            if method not in ("sonic", "density", "neutron"):
                raise ToolError("method must be sonic, density or neutron")
            changes["toc.method"] = method
        final_method = changes.get("toc.method", self.params["toc"]["method"])
        needed = {"sonic": "dt_baseline", "density": "rhob_baseline", "neutron": "nphi_baseline"}[final_method]
        if changes.get(f"toc.{needed}", self.params["toc"].get(needed)) is None:
            raise ToolError(f"the {final_method} method needs {needed}; pass it as an argument")
        return self._what_if(zone, changes, TOC_KEYS + ["phit_mean", "phie_mean"] + SW_KEYS)

    def rank_zones(self, metric, order="high_to_low") -> dict:
        if metric not in RANK_METRICS:
            raise ToolError(f"metric must be one of {RANK_METRICS}")
        have = [(z["averages"][metric], z) for z in self.summary["zones"] if z["averages"][metric] is not None]
        have.sort(key=lambda t: t[0], reverse=(order != "low_to_high"))
        return {"metric": metric, "order": order, "ranking": [
            {"rank": i, "zone": z["name"], metric: value, "low_confidence": z["low_confidence"]}
            for i, (value, z) in enumerate(have, start=1)]}

    # ------------------------------------------------------------------ dispatch
    def dispatch(self, name: str, arguments: dict) -> tuple[dict, bool]:
        """Run a tool by name. Returns (result, is_error)."""
        handlers = {"list_zones": self.list_zones, "get_zone": self.get_zone, "compute_sw": self.compute_sw,
                    "compute_vsh": self.compute_vsh, "compute_porosity": self.compute_porosity,
                    "compute_toc": self.compute_toc, "rank_zones": self.rank_zones}
        try:
            if name not in handlers:
                raise ToolError(f"unknown tool {name!r}")
            result, is_error = handlers[name](**(arguments or {})), False
        except (ToolError, TypeError, ValueError, KeyError) as exc:
            result, is_error = {"error": f"{type(exc).__name__}: {exc}"}, True
        self.calls.append({"tool": name, "input": arguments, "output": result, "is_error": is_error})
        return result, is_error
