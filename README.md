# Shale Log Assistant

An AI log-interpretation assistant for unconventional (shale) reservoirs.

**Rule of the project: Python does every calculation. The LLM never calculates; it only
interprets numbers that Python gives it, and a script checks that afterwards.**

```
LAS file -> QC report -> petrophysics -> facies (random forest) -> zones -> zone_summary.json
                                                                              |
                                        LLM report  <-------------------------+-------> LLM chat with Python tools
                                             |                                               ("what if Rw is 0.05?")
                                        verify_report.py (every number must be in the JSON)
```

## Folder structure

```
shale_log_assistant/
├── config.yaml                  every assumption (matrix density, Rw, a, m, n, baselines, cut-offs ...)
├── requirements.txt
├── README.md
├── shale_assistant/
│   ├── config.py                load + validate config.yaml
│   ├── io_las.py                read LAS (lasio), map mnemonics, convert units
│   ├── qc.py                    depth, missing data, range, flatline, washout, spikes
│   ├── petrophysics.py          Vsh, TOC (Passey), porosity, Sw (Archie, Simandoux) + formulas as text
│   ├── facies.py                random forest, well-based split, confusion matrix, confidence
│   ├── zonation.py              automatic zones + per-zone summary (JSON)
│   ├── pipeline.py              runs everything above in order
│   ├── plotting.py              log plot
│   ├── tools.py                 functions the LLM can call (compute_sw, compute_toc ...)
│   ├── llm.py                   Anthropic API: report writer and chat loop
│   ├── verify.py                number check, 0-1 range check, low-confidence check
│   ├── suggest.py               Rw from Rwa, Passey baseline candidates, matrix density (suggestions only)
│   ├── core_calibration.py      core vs log: bias, RMSE, R2; LOM / baseline change that removes TOC bias
│   └── export.py                HTML + PDF export, verification status at the top
├── scripts/
│   ├── download_example_data.py
│   ├── prepare_force_training.py
│   ├── train_facies.py
│   ├── run_pipeline.py
│   ├── suggest_parameters.py
│   ├── calibrate_with_core.py
│   ├── generate_report.py
│   ├── chat.py
│   ├── verify_report.py
│   ├── csv_to_las.py
│   └── export_report.py
├── tests/                       66 tests, formulas checked against hand calculations
├── models/                      facies metrics + confusion matrix from the example training run
└── outputs/example/             example run on Volve well 15/9-19 SR
```

## How to run

Python 3.10 or newer.

```bash
pip install -r requirements.txt
python -m pytest tests -q                        # 66 tests should pass

python scripts/download_example_data.py          # LAS (2.7 MB) + FORCE 2020 wells (92 MB)
python scripts/prepare_force_training.py         # builds data/force/training_table.csv
python scripts/train_facies.py --compare-random-split     # about 2 minutes on 2 cores
python scripts/run_pipeline.py                   # QC report first, then all calculations
python scripts/suggest_parameters.py             # suggests Rw, Passey baseline, matrix density (prints only)
python scripts/calibrate_with_core.py --core core.csv --depth-shift 0.5   # core TOC/PHI vs logs (prints only)

export ANTHROPIC_API_KEY=sk-ant-...              # PowerShell: $env:ANTHROPIC_API_KEY="sk-ant-..."
python scripts/generate_report.py                # LLM report + automatic verification
python scripts/verify_report.py                  # re-run the verification any time
python scripts/chat.py                           # ask "what if Rw is 0.05 in Z9?"
python scripts/export_report.py                  # one HTML (+ PDF): verification status at the top
```

Your own well: `python scripts/run_pipeline.py --las my_well.las --out outputs/my_well`, then pass
`--out outputs/my_well` to the other scripts. If a curve is not found, add its mnemonic to `curves:` in `config.yaml`.

Test a tool without the LLM (no API key needed):

```bash
python scripts/chat.py --no-llm '{"tool": "compute_sw", "input": {"zone": "Z9", "rw": 0.05}}'
```

## Training the facies classifier on your own labelled wells

```bash
python scripts/train_facies.py --table my_labelled_wells.csv --out models --compare-random-split
```

The table is a CSV (comma or semicolon separated, column names in any case) with one row per depth sample:

| Column | Content | Unit |
|---|---|---|
| `WELL` | well name; this is what the split is made on | text |
| `DEPTH` | measured depth (not used as a feature) | as in your logs |
| `GR` | gamma ray | API |
| `RHOB` | bulk density | g/cc |
| `NPHI` | neutron porosity | v/v (fraction, not percent) |
| `RT` | deep resistivity | ohm.m |
| `DT` | compressional sonic | us/ft |
| `FACIES` | your label (core / XRD lithofacies), any text | - |

```
WELL,DEPTH,GR,RHOB,NPHI,RT,DT,FACIES
A-1,2500.0,142.3,2.48,0.27,8.4,96.1,Siliceous mudstone
A-1,2500.5,155.0,2.44,0.29,11.2,99.8,Organic-rich mudstone
B-2,2710.0,61.7,2.63,0.08,45.0,62.3,Carbonate
```

- Units must be the ones above, because the pipeline predicts on curves in those units. A table that is
  clearly in kg/m3, percent or us/m is refused with a message; nothing is converted silently.
- Rows with an empty curve, an empty label or RT <= 0 are dropped and counted in the output. Extra columns are ignored.
- Classes with fewer than `facies.min_class_samples` samples (config, 100 by default) are dropped: lower it for a small table.
- **5 or more wells:** whole wells are held out (`facies.test_fraction` of the wells, or the wells you list in
  `facies.test_wells`), with well-grouped cross-validation on the training wells.
- **2 to 4 wells:** the script falls back to **leave-one-well-out** and says so: each well is held out once and
  predicted by a model trained on the others; the scores and confusion matrix pool those predictions and a
  per-well table is printed. The saved model is then trained on all wells, so the scores describe models
  trained on one well fewer.
- **1 well:** refused. The only possible test would be a random row split, which leaks.
- A class that occurs in only one well is reported: it cannot be predicted when that well is held out.

## Formulas (units: GR API, RHOB g/cc, NPHI v/v, RT/Rw/Rsh ohm.m, DT us/ft, TOC wt%)

| Quantity | Formula |
|---|---|
| Gamma-ray index | `IGR = (GR - GR_clean) / (GR_shale - GR_clean)`, clipped 0-1 |
| Vsh linear | `Vsh = IGR` |
| Vsh Larionov (older rocks) | `Vsh = 0.33 * (2^(2*IGR) - 1)` |
| Vsh Larionov (Tertiary) | `Vsh = 0.083 * (2^(3.7*IGR) - 1)` |
| Passey, sonic | `DlogR = log10(RT/RT_base) + 0.02 * (DT - DT_base)` |
| Passey, density | `DlogR = log10(RT/RT_base) - 2.50 * (RHOB - RHOB_base)` |
| Passey, neutron | `DlogR = log10(RT/RT_base) + 4.0 * (NPHI - NPHI_base)` |
| TOC | `TOC = DlogR * 10^(2.297 - 0.1688*LOM)`; negative -> 0; 0 where `Vsh < min_vsh` |
| Density porosity | `PHID = (rho_ma - RHOB) / (rho_ma - rho_fl)` |
| Kerogen volume | `Vker = k * (TOC/100) * RHOB / rho_ker` |
| Total porosity | `PHIT = PHID - Vker * (rho_ma - rho_ker) / (rho_ma - rho_fl)` |
| Effective porosity (density) | `PHIE = PHIT - Vsh * PHID_sh`, `PHID_sh = (rho_ma - rho_sh) / (rho_ma - rho_fl)` |
| Effective porosity (density-neutron) | `PHIE = (PHID_c + PHIN_c) / 2`, with `PHIN_c = NPHI - Vker*HI_ker - Vsh*NPHI_sh` (gas: root-mean-square) |
| Sw Archie | `Sw = ((a*Rw) / (PHI^m * RT))^(1/n)` |
| Sw Simandoux | `1/RT = PHI^m * Sw^n / (a*Rw) + Vsh * Sw / Rsh`, solved for Sw |

Sw is clipped to 0-1 and set to 1 where porosity < `phi_min`. The share of samples where Archie gave
Sw > 1 before clipping is kept per zone, because it shows where Rw, m or the matrix density do not fit.
The formulas actually used in a run are also written into `zone_summary.json`.

## Assumptions

All in `config.yaml`. `auto` means Python picks the value from the data and writes what it picked, and how,
into `zone_summary.json -> parameters.auto_picked`. Replace `auto` with your own number whenever you have one.

## Example run: Volve 15/9-19 SR (3550-4630 m)

Console output: `outputs/example/console_output.txt`. Plot: `outputs/example/log_plot.png`.

QC flagged before any calculation: neutron converted from percent; bit size not in the file (8.5 in assumed);
1226 samples (17.3%) washed out; dead sonic and caliper at the bottom of the hole (4597-4618 m);
131 spike samples; a resistivity gap at 3560-3568 m.

| Zone | Top m | Base m | Vsh | PHIT | PHIE | Sw Archie | Sw Simandoux | TOC wt% | Facies (confidence) | Flag |
|---|---|---|---|---|---|---|---|---|---|---|
| Z1 | 3550.2 | 3566.8 | 0.29 | 0.298 | 0.274 | 0.48 | 0.49 | 0.00 | Limestone (0.34) | low confidence |
| Z2 | 3566.8 | 3677.3 | 0.54 | 0.210 | 0.171 | 0.80 | 0.84 | 1.63 | Sandstone (0.70) | low confidence |
| Z3 | 3677.3 | 3821.5 | 0.28 | 0.279 | 0.256 | 0.79 | 0.82 | 0.22 | Sandstone (0.88) | |
| Z4 | 3821.5 | 4119.7 | 0.04 | 0.044 | 0.041 | 0.99 | 0.99 | 0.01 | Limestone (0.67) | low confidence |
| Z5 | 4119.7 | 4270.8 | 0.25 | 0.020 | 0.006 | 1.00 | 0.99 | 0.21 | Limestone (0.58) | low confidence |
| Z6 | 4270.8 | 4304.4 | 0.68 | 0.022 | 0.000 | 1.00 | 1.00 | 1.22 | Limestone (0.43) | low confidence |
| Z7 | 4304.4 | 4310.4 | 0.99 | 0.129 | 0.056 | 0.64 | 0.53 | 4.55 | Shale (0.90) | |
| Z8 | 4310.4 | 4316.3 | 0.88 | 0.024 | 0.000 | 1.00 | 1.00 | 1.28 | Shale (0.69) | low confidence |
| Z9 | 4316.3 | 4341.8 | 0.26 | 0.212 | 0.194 | 0.20 | 0.17 | 0.66 | Sandstone (0.52) | low confidence |
| Z10 | 4341.8 | 4629.7 | 0.62 | 0.118 | 0.070 | 0.98 | 0.98 | 0.07 | Sandstone (0.51) | low confidence |

Z7 is the organic-rich shale: 5.9 m with gamma ray about 250 API and mean TOC 4.55 wt% under the example
assumptions. Z9 is a hydrocarbon-bearing sandstone directly below it.

What-if on Z9, Rw 0.02 -> 0.05 ohm.m (Python tool output): Archie Sw 0.2 -> 0.289, Simandoux Sw 0.175 -> 0.217.

Facies classifier (26 training wells, 9 held-out wells): accuracy 0.642, balanced accuracy 0.469, macro F1 0.336.
A random row split of the same data scores 0.77 accuracy, which is the leakage a well-based split avoids.
Shale is recognised well (F1 0.816); the rare classes are not.

## Second example: USGS GC-1 West Woodway (Eagle Ford Group, thermally immature)

Source: Burke, L.A., Birdwell, J.E., Paxton, S.T., 2022, *Wireline geophysical logging curves and infrared
spectral data for the USGS Gulf Coast #1 West Woodway research wellbore, McLennan County, Texas*,
U.S. Geological Survey data release, doi:10.5066/P9KZVG7P (ScienceBase item 5f4555cc82ce4c3d1225195e).
USGS data release, US Government work (public domain in the US); metadata use constraints: "None",
with the usual USGS no-warranty / no-liability statement.

The release publishes the logs as CSV; `scripts/csv_to_las.py` converts them to LAS without changing a value
(round-trip checked). Run with `config_shale.yaml`, where every changed assumption is marked `[GC-1 CHANGED]`:

```bash
python scripts/csv_to_las.py --csv data/usgs_gc1_west_woodway/wireline_geophysical_welllogs.csv \
    --dictionary data/usgs_gc1_west_woodway/wireline_geophysical_welllogs_datadictionary.csv \
    --out data/usgs_gc1_west_woodway/GC-1_West_Woodway.las --well "USGS GC-1 West Woodway"
python scripts/run_pipeline.py --las data/usgs_gc1_west_woodway/GC-1_West_Woodway.las \
    --config config_shale.yaml --out outputs/gc1
```

Nothing in `outputs/gc1` is calibrated against core. Core geochemistry for this well exists
(USGS data release doi:10.5066/P95QUX1H) and is the next step (`scripts/calibrate_with_core.py`).

## Limits you should state when you present this

- **Not a shale play.** Volve is a conventional North Sea field. It was used because it is public and has a
  full log suite with an organic-rich shale interval. Run a real shale well before making claims about one.
- **The facies labels are lithology classes**, not shale lithofacies from core or XRD. The training code is
  the same; the labels need replacing for a real shale study.
- **No calibration.** Rw, m, n, LOM, kerogen density, matrix density and the baselines are assumed, not
  measured. TOC and Sw are estimates until checked against core TOC, Ro/Tmax and a water sample.
- **One Passey baseline for 1080 m** is too coarse: the baseline shifts with compaction. The 1.63 wt% in Z2
  (shallow, washed out, slow sonic) is most likely a baseline artefact, not kerogen.
- **Matrix density 2.65 g/cc is wrong in the carbonate zones** (Z4-Z6). The "Sw above 1" flag catches it.
- **Archie and Simandoux are conventional-reservoir models.** In shale they are used for lack of better;
  results depend heavily on Rw and m.
- **The LLM report and chat were not run against the live API when this was built** (no API key in the build
  environment). The API calls, tool loop and verification were tested with a scripted fake client.

## Data sources and licences

- Volve well 15/9-19 SR: Equinor Volve open data set, LAS copy from the Petrophysics-Python-Series repository.
- FORCE 2020 lithology competition wells (Bormann et al., 2020).

Both have their own licence terms. Check them before redistributing data or derived files.
