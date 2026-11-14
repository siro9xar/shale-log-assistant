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

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", choices=list(FILES), help="download just one of the two files")
    args = ap.parse_args()
    for key, (url, dest) in FILES.items():
        if args.only and key != args.only:
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists() and dest.stat().st_size > 0:
            print(f"already there: {dest}")
            continue
        print(f"downloading {url}\n         -> {dest}")
        urllib.request.urlretrieve(url, dest)
        print(f"  done, {dest.stat().st_size / 1e6:.1f} MB")
