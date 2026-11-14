"""Hand-calculated checks of every formula (units: g/cc, v/v, ohm.m, us/ft, wt%)."""
import numpy as np
import pytest

from shale_assistant import petrophysics as pp


def test_igr_and_vsh():
    igr = pp.gamma_ray_index(np.array([20.0, 70.0, 120.0, 300.0, 5.0]), 20.0, 120.0)
    assert igr.tolist() == [0.0, 0.5, 1.0, 1.0, 0.0]
    assert pp.vsh_larionov_older(np.array([0.5]))[0] == pytest.approx(0.33, abs=1e-12)       # 0.33*(2^1-1)
    assert pp.vsh_larionov_older(np.array([1.0]))[0] == pytest.approx(0.99, abs=1e-12)       # 0.33*(4-1)
    assert pp.vsh_larionov_tertiary(np.array([1.0]))[0] == pytest.approx(0.083 * (2 ** 3.7 - 1))
    assert pp.vsh_larionov_tertiary(np.array([0.5]))[0] == pytest.approx(0.2162, abs=1e-4)
    with pytest.raises(ValueError):
        pp.gamma_ray_index(np.array([50.0]), 100.0, 20.0)


def test_density_porosity():
    assert pp.density_porosity(np.array([2.40]), 2.65, 1.0)[0] == pytest.approx(0.151515, abs=1e-6)
    assert pp.density_porosity(np.array([2.65]), 2.65, 1.0)[0] == pytest.approx(0.0)


def test_archie():
    # Sw = sqrt(1*0.05 / (0.2^2 * 10)) = sqrt(0.125) = 0.353553
    assert pp.sw_archie(np.array([10.0]), np.array([0.2]), 0.05)[0] == pytest.approx(0.353553, abs=1e-6)
    # m = 1.8, n = 2.2 : (0.05 / (0.2^1.8 * 10))^(1/2.2)
    expected = (0.05 / (0.2 ** 1.8 * 10.0)) ** (1 / 2.2)
    assert pp.sw_archie(np.array([10.0]), np.array([0.2]), 0.05, 1.0, 1.8, 2.2)[0] == pytest.approx(expected)


def test_simandoux():
    rt, phi, rw, rsh = np.array([10.0]), np.array([0.2]), 0.05, 4.0
    # Vsh = 0 reduces to Archie
    assert pp.sw_simandoux(rt, phi, np.array([0.0]), rw, rsh)[0] == pytest.approx(0.353553, abs=1e-6)
    # Vsh = 0.3: C = 0.04/0.05 = 0.8, B = 0.075, D = 0.1 -> Sw = (-0.075 + sqrt(0.075^2 + 0.32)) / 1.6
    sw = pp.sw_simandoux(rt, phi, np.array([0.3]), rw, rsh)[0]
    assert sw == pytest.approx(0.309772, abs=1e-6)
    assert 0.8 * sw ** 2 + 0.075 * sw == pytest.approx(0.1)          # satisfies the Simandoux equation
    # bisection branch (n != 2) satisfies the equation and agrees with the closed form as n -> 2
    sw_n = pp.sw_simandoux(rt, phi, np.array([0.3]), rw, rsh, 1.0, 2.0, 2.2)[0]
    assert 0.8 * sw_n ** 2.2 + 0.075 * sw_n == pytest.approx(0.1, abs=1e-9)
    assert pp.sw_simandoux(rt, phi, np.array([0.3]), rw, rsh, 1.0, 2.0, 2.0000001)[0] == pytest.approx(sw, abs=1e-6)


def test_passey():
    # sonic: log10(10/2) + 0.02*(100-80) = 0.69897 + 0.4 = 1.09897
    d = pp.delta_log_r(np.array([10.0]), 2.0, np.array([100.0]), 80.0, "sonic")[0]
    assert d == pytest.approx(1.09897, abs=1e-5)
    # LOM 10: 10^(2.297 - 1.688) = 10^0.609 = 4.0644 -> TOC = 4.4667 wt%
    assert pp.toc_passey(np.array([d]), 10.0)[0] == pytest.approx(4.4667, abs=1e-3)
    # density: log10(10/2) - 2.5*(2.40-2.55) = 0.69897 + 0.375
    assert pp.delta_log_r(np.array([10.0]), 2.0, np.array([2.40]), 2.55, "density")[0] == pytest.approx(1.07397, abs=1e-5)
    # neutron: log10(10/2) + 4*(0.30-0.20) = 1.09897
    assert pp.delta_log_r(np.array([10.0]), 2.0, np.array([0.30]), 0.20, "neutron")[0] == pytest.approx(1.09897, abs=1e-5)
    assert pp.toc_passey(np.array([-0.5]), 10.0)[0] == 0.0           # negative separation -> 0, never negative TOC


def test_kerogen_volume_and_correction():
    # 5 wt% TOC, k = 1.2, RHOB 2.4, rho_ker 1.3 -> 1.2 * 0.05 * 2.4 / 1.3 = 0.110769
    vk = pp.kerogen_volume(np.array([5.0]), np.array([2.4]), 1.3, 1.2)[0]
    assert vk == pytest.approx(0.110769, abs=1e-6)
    # volumetric balance: rho_b = rho_ma*(1-phi-Vk) + rho_k*Vk + rho_f*phi must return the corrected porosity
    phi_true = 0.06
    rhob = 2.65 * (1 - phi_true - vk) + 1.3 * vk + 1.0 * phi_true
    vk2 = pp.kerogen_volume(np.array([5.0]), np.array([rhob]), 1.3, 1.2)[0]
    phid = pp.density_porosity(np.array([rhob]), 2.65, 1.0)[0]
    assert phid - vk2 * (2.65 - 1.3) / (2.65 - 1.0) == pytest.approx(
        phi_true + (vk - vk2) * (2.65 - 1.3) / 1.65, abs=1e-9)


PARAMS = {
    "vsh": {"gr_curve": "GR", "gr_clean": 20.0, "gr_shale": 120.0, "larionov_variant": "older",
            "use_for_downstream": "larionov"},
    "toc": {"method": "sonic", "lom": 10.0, "min_vsh": 0.4, "rt_baseline": 2.0, "dt_baseline": 80.0,
            "rhob_baseline": None, "nphi_baseline": None},
    "porosity": {"rho_matrix": 2.65, "rho_fluid": 1.0, "rho_shale": 2.55, "nphi_shale": 0.3,
                 "nphi_matrix_offset": 0.0, "fluid_type": "liquid", "phit_method": "density",
                 "phie_method": "density", "phi_max": 0.45,
                 "kerogen": {"apply": True, "rho_kerogen": 1.3, "k_factor": 1.2, "hi_kerogen": 0.65}},
    "sw": {"a": 1.0, "m": 2.0, "n": 2.0, "rw": 0.05, "rsh": 4.0, "archie_porosity": "PHIT",
           "simandoux_porosity": "PHIE", "phi_min": 0.01},
}


def test_compute_all_ranges_and_nan_handling():
    import pandas as pd
    rng = np.random.default_rng(0)
    n = 500
    df = pd.DataFrame({"GR": rng.uniform(0, 300, n), "RHOB": rng.uniform(1.9, 3.0, n),
                       "NPHI": rng.uniform(-0.05, 0.7, n), "RT": 10 ** rng.uniform(-1, 3, n),
                       "DT": rng.uniform(45, 150, n)})
    df.loc[::17, "RHOB"] = np.nan
    df.loc[::23, "RT"] = np.nan
    out = pp.compute_all(df, PARAMS)
    for col in ("VSH_LINEAR", "VSH_LARIONOV", "VSH", "SW_ARCHIE", "SW_SIMANDOUX"):
        x = out[col].dropna()
        assert len(x) and x.min() >= 0.0 and x.max() <= 1.0, col
    for col in ("PHIT", "PHIE"):
        x = out[col].dropna()
        assert x.min() >= 0.0 and x.max() <= 0.45
    assert (out["PHIE"].dropna() <= out["PHIT"].dropna() + 1e-12).all()
    assert out["TOC"].dropna().min() >= 0.0
    assert out.loc[::17, "PHIT"].isna().all()                 # no density -> no porosity, not a made-up value
    assert out["SW_ARCHIE"][df["RT"].isna()].isna().all()     # no resistivity -> no Sw


def test_single_sample_end_to_end():
    import pandas as pd
    df = pd.DataFrame({"GR": [70.0], "RHOB": [2.40], "NPHI": [0.25], "RT": [10.0], "DT": [100.0]})
    out = pp.compute_all(df, PARAMS).iloc[0]
    assert out["VSH"] == pytest.approx(0.33)
    assert out["TOC"] == 0.0                                   # Vsh 0.33 < min_vsh 0.4 -> reservoir rock, TOC masked
    assert out["PHIT"] == pytest.approx(0.151515, abs=1e-6)
    assert out["PHIE"] == pytest.approx(0.151515 - 0.33 * (0.10 / 1.65), abs=1e-6)
    assert out["SW_ARCHIE"] == pytest.approx((0.05 / (0.151515 ** 2 * 10)) ** 0.5, abs=1e-5)
