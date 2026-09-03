#!/usr/bin/env python3
import argparse, json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/"src"))
from spatial_rg.extraction import run
if __name__=="__main__":
 p=argparse.ArgumentParser(); p.add_argument("--device",default="cuda"); p.add_argument("--batch-size",type=int,default=64); p.add_argument("--limit",type=int); p.add_argument("--num-shards",type=int,default=1); p.add_argument("--shard-index",type=int,default=0); p.add_argument("--overwrite",action="store_true"); p.add_argument("--skip-source-hash",action="store_true"); a=p.parse_args(); print(json.dumps(run(a),indent=2))
