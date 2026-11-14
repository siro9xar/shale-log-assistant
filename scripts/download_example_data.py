"""Download the public example data (run once).

1. Volve well 15/9-19 SR composite log (LAS)  -> data/volve/15-9-19_SR_COMP.LAS
   Equinor Volve open data set, copy hosted in A. McDonald's Petrophysics-Python-Series repository.
2. FORCE 2020 lithology competition training wells -> data/force/train.zip  (about 92 MB)
   98 labelled Norwegian wells, used to train the facies classifier.

Both data sets have their own licence terms: read them before you redistribute the files.
"""
import argparse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FILES = {
    "las": ("https://raw.githubusercontent.com/andymcdgeo/Petrophysics-Python-Series/master/Data/15-9-19_SR_COMP.LAS",
            ROOT / "data" / "volve" / "15-9-19_SR_COMP.LAS"),
    "force": ("https://raw.githubusercontent.com/bolgebrygg/Force-2020-Machine-Learning-competition/master/"
              "lithology_competition/data/train.zip", ROOT / "data" / "force" / "train.zip"),
}
# Optional (python scripts/download_example_data.py --only gc1): USGS GC-1 West Woodway, Eagle Ford Group.
# USGS data release doi:10.5066/P9KZVG7P (public domain). CSV; convert with scripts/csv_to_las.py.
_SB = "https://www.sciencebase.gov/catalog/file/get/5f4555cc82ce4c3d1225195e?f=__disk__"
_GC1 = ROOT / "data" / "usgs_gc1_west_woodway"
OPTIONAL = {"gc1": [
    (_SB + "83%2Fa9%2Fd5%2F83a9d5f0742d8ad74cfd2736b151605df83d081c", _GC1 / "wireline_geophysical_welllogs.csv"),
    (_SB + "11%2Faa%2Fd5%2F11aad55a7a00149e8e871c08ba8ca1af01382be6",
     _GC1 / "wireline_geophysical_welllogs_datadictionary.csv"),
    (_SB + "ef%2F86%2F52%2Fef8652ea722f08f639c91351fdfff2e1dfab98cc", _GC1 / "P9KZVG7P_metadata.xml"),
]}

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", choices=list(FILES) + list(OPTIONAL),
                    help="download just this data set (gc1 is optional and only downloaded when named here)")
    args = ap.parse_args()
    todo = [(k, u, d) for k, (u, d) in FILES.items()] + [(k, u, d) for k, items in OPTIONAL.items() for u, d in items]
    for key, url, dest in todo:
        if (args.only and key != args.only) or (not args.only and key in OPTIONAL):
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists() and dest.stat().st_size > 0:
            print(f"already there: {dest}")
            continue
        print(f"downloading {url}\n         -> {dest}")
        urllib.request.urlretrieve(url, dest)
        print(f"  done, {dest.stat().st_size / 1e6:.1f} MB")
