#!/usr/bin/env python3
from pathlib import Path
import sys, json
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/"src"))
from spatial_rg.targets import build_all
if __name__=="__main__": print(json.dumps(build_all(),indent=2))
