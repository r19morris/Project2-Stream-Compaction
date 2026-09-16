import subprocess, re, matplotlib.pyplot as plt
from collections import defaultdict

exe = "build/bin/Release/cis5650_stream_compaction_test.exe"
data = defaultdict(list)
exps = range(8, 25, 2)

for e in exps:
    out = subprocess.run([exe, str(e)], capture_output=True, text=True).stdout
    for name, t in re.findall(r"==== (.+?) ====\s+elapsed time: ([\d.]+)ms", out):
        if "compact" in name:
            continue
        data[name].append(t and float(t))

for name, ts in data.items():
    if "scan, non-power-of-two" in name and "compact" not in name:
        plt.plot([2**e for e in exps], ts, marker="o", label=name)
plt.xscale("log", base=2); plt.yscale("log")
plt.xlabel("Array size"); plt.ylabel("Time (ms)"); plt.legend(); plt.show()