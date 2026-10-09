"""Spatiotemporal blue noise by void and cluster (after Wolfe et al. 2022): a W x W x T stack of masks in which every
2D slice is blue noise and every pixel's sequence through the slices is blue noise too.

Energy between two pixels (toroidal distances): mode 'cross' counts pairs in the same slice by a spatial Gaussian and
pairs at the same pixel by a temporal Gaussian; mode 'product' uses exp(-ds^2/2ss^2) * exp(-dt^2/2st^2) for all pairs.
Writes a uint8 array (T, W, W) of ranks scaled to 0..255.

Usage: python stbn_gen.py out.bin [W] [T] [mode] [sigma_s] [sigma_t] [seed]
"""
import sys, time
import numpy as np

out = sys.argv[1]
W = int(sys.argv[2]) if len(sys.argv) > 2 else 64
T = int(sys.argv[3]) if len(sys.argv) > 3 else 32
mode = sys.argv[4] if len(sys.argv) > 4 else 'cross'
ss = float(sys.argv[5]) if len(sys.argv) > 5 else 1.9
st = float(sys.argv[6]) if len(sys.argv) > 6 else 1.9
seed = int(sys.argv[7]) if len(sys.argv) > 7 else 7
rng = np.random.default_rng(seed)
N = T * W * W

# Kernels as offset lists: (dt, dy, dx, weight).
R = int(np.ceil(3 * ss))
offs = []
if mode == 'cross':
    for dy in range(-R, R + 1):
        for dx in range(-R, R + 1):
            offs.append((0, dy, dx, np.exp(-(dx * dx + dy * dy) / (2 * ss * ss))))
    for dt in range(1, T):
        d = min(dt, T - dt)
        offs.append((dt, 0, 0, np.exp(-(d * d) / (2 * st * st))))
else:
    Rt = min(int(np.ceil(3 * st)), T // 2)
    for dt in range(-Rt, Rt + 1):
        for dy in range(-R, R + 1):
            for dx in range(-R, R + 1):
                offs.append((dt, dy, dx, np.exp(-(dx * dx + dy * dy) / (2 * ss * ss)) * np.exp(-(dt * dt) / (2 * st * st))))
offs = np.array(offs)
odt, ody, odx, ow = offs[:, 0].astype(int), offs[:, 1].astype(int), offs[:, 2].astype(int), offs[:, 3]


def neighbours(i):
    t, r = divmod(i, W * W)
    y, x = divmod(r, W)
    return ((t + odt) % T) * W * W + ((y + ody) % W) * W + (x + odx) % W


energy = np.zeros(N)
on = np.zeros(N, dtype=bool)


def toggle(i, sign):
    np.add.at(energy, neighbours(i), sign * ow)


# Initial pattern: 10% of pixels, at random, then relaxed by moving the tightest cluster to the largest void.
start = N // 10
for i in rng.choice(N, start, replace=False):
    on[i] = True
    toggle(i, 1.0)
t0 = time.time()
while True:
    e_on = np.where(on, energy, -np.inf)
    c = int(np.argmax(e_on))
    on[c] = False
    toggle(c, -1.0)
    e_off = np.where(on, np.inf, energy)
    v = int(np.argmin(e_off))
    if v == c:
        on[c] = True
        toggle(c, 1.0)
        break
    on[v] = True
    toggle(v, 1.0)
print('initial pattern relaxed', round(time.time() - t0, 1), 's')

rank = np.zeros(N, dtype=np.int64)
proto = on.copy()
proto_energy = energy.copy()
# Phase 1: remove the tightest clusters, ranks start-1 down to 0.
for r in range(start - 1, -1, -1):
    e_on = np.where(on, energy, -np.inf)
    c = int(np.argmax(e_on))
    on[c] = False
    toggle(c, -1.0)
    rank[c] = r
# Phases 2 and 3: from the prototype, fill the largest voids (with a symmetric kernel the tightest cluster of
# zeros is the largest void of ones), ranks start up to N-1.
on[:] = proto
energy[:] = proto_energy
t0 = time.time()
for r in range(start, N):
    e_off = np.where(on, np.inf, energy)
    v = int(np.argmin(e_off))
    on[v] = True
    toggle(v, 1.0)
    rank[v] = r
    if r % 20000 == 0:
        print(r, '/', N, round(time.time() - t0, 1), 's', flush=True)
vals = np.floor((rank + 0.5) * 256 / N).astype(np.uint8).reshape(T, W, W)
vals.tofile(out)
print('wrote', out, vals.shape)
