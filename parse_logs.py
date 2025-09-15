#!/usr/bin/env python3
"""
parse_logs.py - Extract structured training metrics from BitterBot logs

Usage:
    python parse_logs.py path/to/train.log
"""

import re
import sys
from collections import defaultdict

patterns = {
    "breakthrough": re.compile(r"\[Breakthrough\] EM=(\d+\.\d+)% at step=(\d+)"),
    "dopamine": re.compile(r"\[Dopamine\] Stored (\d+) puzzle pair"),
    "dopamine_skip": re.compile(r"\[Dopamine\] capture pipeline skipped: (.+)"),
    "buffer_size": re.compile(r"buffer size=(\d+)"),
    "nightmare": re.compile(r"\[Nightmare\] Applied negative replay on (\d+)/(\d+) failures at step (\d+)"),
    "relmem": re.compile(r"\[Step (\d+)\] RelMem: (.+)"),
    "dream_tokens": re.compile(r"\[Dream\] _dream_tokens shape: (.+)"),
    "dream_motif": re.compile(r"motifs_added=(\d+)"),
    "wormhole_mine": re.compile(r"\[Wormhole\] mined (\d+) templates"),
    "wormhole_cons": re.compile(r"\[Wormhole\] Consolidated (\d+) templates"),
    "phi": re.compile(r"\[EBR\] phi=([\d\.eE+-]+)"),
    "kappa": re.compile(r"\[EBR\] kappa=([\d\.eE+-]+)"),
    "cge": re.compile(r"\[EBR\] cge=([\d\.eE+-]+)")
}

def parse_log(path):
    data = defaultdict(list)
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            for key, pat in patterns.items():
                m = pat.search(line)
                if m:
                    data[key].append(m.groups())
    return data

def summarize(data):
    print("=== Structured Log Summary ===")
    print("\n🎯 Breakthroughs")
    for em, step in data["breakthrough"]:
        print(f" - Step {step}: EM={em}%")

    print("\n💊 Dopamine")
    for pairs, in data["dopamine"]:
        print(f" - Stored {pairs} pair(s)")
    if data["dopamine_skip"]:
        print(" - Skipped captures:")
        for reason, in data["dopamine_skip"]:
            print(f"   {reason}")

    print("\n🌑 Nightmares")
    for applied, total, step in data["nightmare"]:
        print(f" - Step {step}: pruned {applied}/{total} failures")

    print("\n🧠 RelMem")
    for step, stats in data["relmem"]:
        print(f" - Step {step}: {stats}")

    print("\n🌙 Dream Engine")
    for shape, in data["dream_tokens"]:
        print(f" - Dream tokens {shape}")
    for motifs, in data["dream_motif"]:
        print(f" - Motifs added {motifs}")

    print("\n🌀 Wormhole")
    for count, in data["wormhole_mine"]:
        print(f" - Mined {count} templates")
    for count, in data["wormhole_cons"]:
        print(f" - Consolidated {count} templates")

    print("\nΦ / κ / CGE Priors")
    for phi in data["phi"]:
        print(f" - phi={phi[0]}")
    for kappa in data["kappa"]:
        print(f" - kappa={kappa[0]}")
    for cge in data["cge"]:
        print(f" - cge={cge[0]}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python parse_logs.py path/to/train.log")
        sys.exit(1)
    path = sys.argv[1]
    data = parse_log(path)
    summarize(data)