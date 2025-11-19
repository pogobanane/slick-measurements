import os
import importlib
import sys

def foobar():
    print("ok7")

SIZE_SMALL = 64
SIZE_BIG = 1522

def ns2s(ns) -> float:
    return ns / 1e9

def s2ns(s) -> float:
    return s * 1e9

def pps2mpps(pps) -> float:
    return pps / 1e6

def mpps2pps(mpps) -> float:
    return mpps * 1e6

def mpps2gbit(mpps, size = SIZE_SMALL):
    bitps = mpps2pps(mpps) * (size + 20) * 8
    return bitps / 1e9

def mpps2nspp(mpps) -> float:
    return s2ns(1 / mpps2pps(mpps))

def nspp2mpps(nspp) -> float:
    return pps2mpps(1 / ns2s(nspp))

def gbit2mpps(gbitps, size = SIZE_SMALL):
    pps = gbitps * 1e9 / ((size + 20) * 8)
    return pps2mpps(pps)

def gb2gbit(gb):
    return gb * 8

def cycles2ns(cycles, freq_mhz=1996):
    s = cycles / (1e6*freq_mhz)
    return s * 1e9

print("Repl.py loaded 🦘")
print("Reload with `repl.reload()`")


# if __name__ == "__main__":
    # print("This script is supposed to be used for static definitions in `make repl`")
    # os.exit(1)
