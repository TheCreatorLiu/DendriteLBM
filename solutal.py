"""Solutal dendritic solidification, manuscript section 4.2.

Run `python solutal.py --help`. Numerical provenance and deliberate changes:
docs/MANUSCRIPT_ALIGNMENT.md. Each invocation is an isolated simulation process.
"""
import math
from runtime import configure, Session

CFG = configure("solutal")
import warp as wp
wp.init()
if CFG.device.startswith("cuda") and not wp.is_cuda_available():
    raise RuntimeError("CUDA requested but unavailable; check the NVIDIA driver or use --device cpu")
wp.set_device(CFG.device)
DEVICE = str(wp.get_device())
FLOW = CFG.flow == "on"

dtype = wp.float32

# ============================================================
# Constants and lattice
# ============================================================
NX = CFG.nx
NY = CFG.ny
Q = 9
PI = math.pi

dim_space = (NX, NY)
dim_u = (NX, NY, 2)
dim_f = (NX, NY, Q)

vec9f = wp.types.vector(length=Q, dtype=dtype)
vec9i = wp.types.vector(length=Q, dtype=wp.int32)
mat92f = wp.types.matrix(shape=(Q, 2), dtype=dtype)
mat99f = wp.types.matrix(shape=(Q, Q), dtype=dtype)

w = wp.constant(vec9f(
    4.0 / 9.0,
    1.0 / 9.0,
    1.0 / 9.0,
    1.0 / 9.0,
    1.0 / 9.0,
    1.0 / 36.0,
    1.0 / 36.0,
    1.0 / 36.0,
    1.0 / 36.0,
))

ei = wp.constant(wp.types.matrix(shape=(Q, 2), dtype=wp.int32)([
    [0, 0], [1, 0], [0, 1], [-1, 0], [0, -1], [1, 1], [-1, 1], [-1, -1], [1, -1]
]))

ed = wp.constant(mat92f([
    [0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0],
    [1.0, 1.0], [-1.0, 1.0], [-1.0, -1.0], [1.0, -1.0]
]))

invk = wp.constant(vec9i([0, 3, 4, 1, 2, 7, 8, 5, 6]))

M = wp.constant(mat99f([
    [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    [-4.0, -1.0, -1.0, -1.0, -1.0, 2.0, 2.0, 2.0, 2.0],
    [4.0, -2.0, -2.0, -2.0, -2.0, 1.0, 1.0, 1.0, 1.0],
    [0.0, 1.0, 0.0, -1.0, 0.0, 1.0, -1.0, -1.0, 1.0],
    [0.0, -2.0, 0.0, 2.0, 0.0, 1.0, -1.0, -1.0, 1.0],
    [0.0, 0.0, 1.0, 0.0, -1.0, 1.0, 1.0, -1.0, -1.0],
    [0.0, 0.0, -2.0, 0.0, 2.0, 1.0, 1.0, -1.0, -1.0],
    [0.0, 1.0, -1.0, 1.0, -1.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, -1.0, 1.0, -1.0],
]))

invM = wp.constant(mat99f([
    [4.0, -4.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [4.0, -1.0, -2.0, 6.0, -6.0, 0.0, 0.0, 9.0, 0.0],
    [4.0, -1.0, -2.0, 0.0, 0.0, 6.0, -6.0, -9.0, 0.0],
    [4.0, -1.0, -2.0, -6.0, 6.0, 0.0, 0.0, 9.0, 0.0],
    [4.0, -1.0, -2.0, 0.0, 0.0, -6.0, 6.0, -9.0, 0.0],
    [4.0, 2.0, 1.0, 6.0, 3.0, 6.0, 3.0, 0.0, 9.0],
    [4.0, 2.0, 1.0, -6.0, -3.0, 6.0, 3.0, 0.0, -9.0],
    [4.0, 2.0, 1.0, -6.0, -3.0, -6.0, -3.0, 0.0, 9.0],
    [4.0, 2.0, 1.0, 6.0, 3.0, -6.0, -3.0, 0.0, -9.0],
]))

# ============================================================
# Physical parameters (see manuscript alignment notes)
# ============================================================
dx = 1.0
nF = CFG.nF
nG = CFG.nG
nT = CFG.nT
nH = CFG.nH

mF = int(round(nF / nF))
mG = int(round(nG / nF))
mT = int(round(nT / nF))
mH = int(round(nH / nF))

dtF = nF * dx * dx
dtG = nG * dx * dx
dtT = nT * dx * dx
dtH = nH * dx * dx
cF = dx / dtF
cG = dx / dtG
cT = dx / dtT
cH = dx / dtH

Pe = 0.25
Pr = 23.1
Le = 1.0
W0 = 2.5 * dx
tau0 = 50.0 * dtG
eps = 0.02
d0 = 0.2762 * W0
alpha = 4.0 * W0 * W0 / tau0
lamda = 0.8839 * W0 / d0
MC = 1.0
U0 = -0.55
k0 = 0.15
DL = 0.25
DS = DL * 1e-2
CL = 0.1
Ceq = 0.18779
CPL = 1.0
CPS = 1.0
Latent = 1.0
kL = alpha
kS = alpha
Tm = 0.0
mm = 0.0
T0 = 0.0 * Latent / CPL
niuL = Pr * alpha
Gx = 0.0
Gy = 0.0

# ============================================================
# Dendrite-tip velocity statistics
# ============================================================
# psi = TIP_PSI_THRESHOLD is regarded as the solid-liquid interface.
# The four primary tips are tracked along the center lines of the seed.
TIP_PSI_THRESHOLD = 0.0
TIP_CENTER_X = NX // 2
TIP_CENTER_Y = NY // 2
# 1500 base steps = 100 phase-field time units because dtF=1/15 and mG=15.
# Keeping this value as an integer multiple of mG avoids sampling between G updates.
TIP_VELOCITY_EVERY = 1000

INLET_UX = (W0 / tau0 / 1.0) if FLOW else 0.0

# ============================================================
# Arrays
# ============================================================
space = wp.zeros(dim_space, dtype=wp.int32)

# double-buffer distributions: lowercase=current, uppercase=post-stream buffer
f = wp.zeros(dim_f, dtype=dtype)
F = wp.zeros(dim_f, dtype=dtype)
g = wp.zeros(dim_f, dtype=dtype)
G = wp.zeros(dim_f, dtype=dtype)
t = wp.zeros(dim_f, dtype=dtype)
T = wp.zeros(dim_f, dtype=dtype)
h = wp.zeros(dim_f, dtype=dtype)
H = wp.zeros(dim_f, dtype=dtype)

rho = wp.zeros(dim_space, dtype=dtype)
niu = wp.zeros(dim_space, dtype=dtype)
Temp = wp.zeros(dim_space, dtype=dtype)
psi = wp.zeros(dim_space, dtype=dtype)
psi0 = wp.zeros(dim_space, dtype=dtype)
as_field = wp.zeros(dim_space, dtype=dtype)
tauG_field = wp.zeros(dim_space, dtype=dtype)
tauH_field = wp.zeros(dim_space, dtype=dtype)
tauT_field = wp.zeros(dim_space, dtype=dtype)
FU = wp.zeros(dim_space, dtype=dtype)

BF = wp.zeros(dim_u, dtype=dtype)
u = wp.zeros(dim_u, dtype=dtype)
v = wp.zeros(dim_u, dtype=dtype)
norm = wp.zeros(dim_u, dtype=dtype)
Jat = wp.zeros((2, NX, NY), dtype=dtype)

psiAccumT = wp.zeros(dim_space, dtype=dtype)
psiAccumH = wp.zeros(dim_space, dtype=dtype)
JatDivAccumH = wp.zeros(dim_space, dtype=dtype)

Grad_psi = wp.zeros(dim_u, dtype=dtype)
Grad_UC = wp.zeros(dim_u, dtype=dtype)
Grad_Temp = wp.zeros(dim_u, dtype=dtype)

fs = wp.zeros(dim_space, dtype=dtype)
fl = wp.zeros(dim_space, dtype=dtype)
UC = wp.zeros(dim_space, dtype=dtype)
C = wp.zeros(dim_space, dtype=dtype)
QT = wp.zeros(dim_space, dtype=dtype)

# ============================================================
# Helper functions
# ============================================================
SEED_WIDTH = math.sqrt(2.0) * W0 if CFG.parameters == "paper" else math.sqrt(2.0 * W0)

@wp.func
def clamp_i(x: int, a: int, b: int):
    if x < a:
        return a
    if x > b:
        return b
    return x

@wp.func
def grad_scalar(arr: wp.array2d(dtype=dtype), i: int, j: int, xy: int):
    s = float(0.0)
    for k in range(1, Q):
        ip = clamp_i(i + ei[k, 0], 0, NX - 1)
        jp = clamp_i(j + ei[k, 1], 0, NY - 1)
        s += w[k] * arr[ip, jp] * ed[k, xy]
    return 3.0 * s

@wp.func
def Feq(k: int, rho_v: float, ux: float, uy: float):
    eu = (ed[k, 0] * ux + ed[k, 1] * uy) / cF
    uv = (ux * ux + uy * uy) / (cF * cF)
    return w[k] * rho_v * (1.0 + 3.0 * eu + 4.5 * eu * eu - 1.5 * uv)

@wp.func
def Geq(k: int, psi_v: float, vx: float, vy: float):
    ev = (ed[k, 0] * vx + ed[k, 1] * vy) / cG
    return w[k] * (psi_v + 3.0 * ev)

@wp.func
def Teq(k: int, theta: float, ux: float, uy: float):
    eu = (ed[k, 0] * ux + ed[k, 1] * uy) / cT
    return w[k] * theta * (1.0 + 3.0 * eu)

@wp.func
def Heq(k: int, uc: float, ux: float, uy: float):
    eu = (ed[k, 0] * ux + ed[k, 1] * uy) / cH
    return w[k] * uc * (1.0 + 3.0 * eu)

# ============================================================
# Initialization kernels
# ============================================================
@wp.kernel
def init_space_kernel(space: wp.array2d(dtype=wp.int32)):
    i, j = wp.tid()
    if i == 0 or i == NX - 1 or j == 0 or j == NY - 1:
        space[i, j] = 1
    else:
        space[i, j] = 0

@wp.kernel
def initialize_fields(
    space: wp.array2d(dtype=wp.int32),
    rho: wp.array2d(dtype=dtype),
    u: wp.array3d(dtype=dtype),
    v: wp.array3d(dtype=dtype),
    UC: wp.array2d(dtype=dtype),
    Temp: wp.array2d(dtype=dtype),
    psi: wp.array2d(dtype=dtype),
    psi0: wp.array2d(dtype=dtype),
    fs: wp.array2d(dtype=dtype),
    fl: wp.array2d(dtype=dtype),
    FU: wp.array2d(dtype=dtype),
):
    i, j = wp.tid()
    if space[i, j] == 0:
        rho[i, j] = 1.0
        u[i, j, 0] = 0.0
        u[i, j, 1] = 0.0
        v[i, j, 0] = 0.0
        v[i, j, 1] = 0.0
        UC[i, j] = U0
        Temp[i, j] = T0

        r = wp.sqrt((float(i) - float(NX) / 2.0) **2.0 + (float(j) - float(NY) / 2.0) **2.0)
        R0 = 10.0
        val = wp.tanh((R0 - r) / SEED_WIDTH)
        psi[i, j] = val
        psi0[i, j] = val
        fs[i, j] = (psi[i, j] + 1.0) * 0.5
        fl[i, j] = 1.0 - fs[i, j]
        FU[i, j] = 1.0 / Le

@wp.kernel
def initialize_tau_and_eq(
    space: wp.array2d(dtype=wp.int32),
    rho: wp.array2d(dtype=dtype),
    u: wp.array3d(dtype=dtype),
    v: wp.array3d(dtype=dtype),
    UC: wp.array2d(dtype=dtype),
    Temp: wp.array2d(dtype=dtype),
    psi: wp.array2d(dtype=dtype),
    fs: wp.array2d(dtype=dtype),
    FU: wp.array2d(dtype=dtype),
    as_field: wp.array2d(dtype=dtype),
    tauG_field: wp.array2d(dtype=dtype),
    tauH_field: wp.array2d(dtype=dtype),
    tauT_field: wp.array2d(dtype=dtype),
    f: wp.array3d(dtype=dtype),
    g: wp.array3d(dtype=dtype),
    t: wp.array3d(dtype=dtype),
    h: wp.array3d(dtype=dtype),
):
    i, j = wp.tid()
    if space[i, j] == 0:
        phi = psi[i, j]
        fu = FU[i, j]
        uc = UC[i, j]
        gpx = grad_scalar(psi, i, j, 0)
        gpy = grad_scalar(psi, i, j, 1)
        gmode = wp.sqrt(gpx * gpx + gpy * gpy)
        asn = 1.0
        NGx = 0.0
        NGy = 0.0
        if gmode > 1.0e-6:
            gx2 = gpx * gpx
            gy2 = gpy * gpy
            mode4 = gmode * gmode * gmode * gmode
            asn = 1.0 - 3.0 * eps + 4.0 * eps * (gx2 * gx2 + gy2 * gy2) / mode4
            NGx = 16.0 * eps * asn * gpx * gy2 * (gx2 - gy2) / mode4
            NGy = 16.0 * eps * asn * gpy * gx2 * (gy2 - gx2) / mode4
        v[i, j, 0] = -NGx * W0 * W0 / (fu * tau0)
        v[i, j, 1] = -NGy * W0 * W0 / (fu * tau0)
        as_field[i, j] = asn
        tauG_field[i, j] = 3.0 * nG * (asn * asn * W0 * W0 / (fu * tau0)) + 0.5

        k_phi = (1.0 + k0) - (1.0 - k0) * phi
        DLS_phi = (1.0 + phi) * DS + (1.0 - phi) * DL
        Deff = DLS_phi / k_phi
        tauH_field[i, j] = 3.0 * nH * Deff + 0.5

        CPeff = CPS * fs[i, j] + CPL * (1.0 - fs[i, j])
        Xeff = (kS * fs[i, j] + kL * (1.0 - fs[i, j])) / CPeff
        tauT_field[i, j] = 3.0 * nT * Xeff + 0.5

        ux = u[i, j, 0]
        uy = u[i, j, 1]
        vx = v[i, j, 0]
        vy = v[i, j, 1]
        th = Temp[i, j]
        uu = UC[i, j]
        rr = rho[i, j]
        pp = psi[i, j]
        for k in range(Q):
            f[i, j, k] = Feq(k, rr, ux, uy)
            g[i, j, k] = Geq(k, pp, vx, vy)
            t[i, j, k] = Teq(k, th, ux, uy)
            h[i, j, k] = Heq(k, uu, ux, uy)

# ============================================================
# Macro kernels
# ============================================================
@wp.kernel
def macroF_kernel(
    space: wp.array2d(dtype=wp.int32),
    f: wp.array3d(dtype=dtype),
    rho: wp.array2d(dtype=dtype),
    BF: wp.array3d(dtype=dtype),
    u: wp.array3d(dtype=dtype),
):
    i, j = wp.tid()
    if space[i, j] == 0:
        rr = 0.0
        ux = 0.0
        uy = 0.0
        for k in range(Q):
            fk = f[i, j, k]
            rr += fk
            ux += fk * ed[k, 0]
            uy += fk * ed[k, 1]
        rho[i, j] = rr
        u[i, j, 0] = (ux + 0.5 * dtF * BF[i, j, 0]) / rr * cF
        u[i, j, 1] = (uy + 0.5 * dtF * BF[i, j, 1]) / rr * cF

@wp.kernel
def macroG_kernel(
    space: wp.array2d(dtype=wp.int32),
    g: wp.array3d(dtype=dtype),
    psi: wp.array2d(dtype=dtype),
    psi0: wp.array2d(dtype=dtype),
    v: wp.array3d(dtype=dtype),
    tauG_field: wp.array2d(dtype=dtype),
    Grad_psi: wp.array3d(dtype=dtype),
    fs: wp.array2d(dtype=dtype),
    fl: wp.array2d(dtype=dtype),
):
    i, j = wp.tid()
    if space[i, j] == 0:
        psi0[i, j] = psi[i, j]
        p = 0.0
        for k in range(Q):
            p += g[i, j, k]
        psi[i, j] = p
        gx = 0.0
        gy = 0.0
        for k in range(1, Q):
            geq = Geq(k, p, v[i, j, 0], v[i, j, 1])
            gx += (g[i, j, k] - geq) * ed[k, 0]
            gy += (g[i, j, k] - geq) * ed[k, 1]
        denom = -(tauG_field[i, j] * dtG / 3.0)
        Grad_psi[i, j, 0] = gx / denom
        Grad_psi[i, j, 1] = gy / denom
        fs[i, j] = (p + 1.0) * 0.5
        fl[i, j] = 1.0 - fs[i, j]

@wp.kernel
def macroT_kernel(
    space: wp.array2d(dtype=wp.int32),
    t: wp.array3d(dtype=dtype),
    Temp: wp.array2d(dtype=dtype),
    u: wp.array3d(dtype=dtype),
    tauT_field: wp.array2d(dtype=dtype),
    Grad_Temp: wp.array3d(dtype=dtype),
):
    i, j = wp.tid()
    if space[i, j] == 0:
        TT0 = 0.0
        for k in range(Q):
            TT0 += t[i, j, k]
        Temp[i, j] = TT0
        gx = 0.0
        gy = 0.0
        for k in range(1, Q):
            teq = Teq(k, TT0, u[i, j, 0], u[i, j, 1])
            gx += (t[i, j, k] - teq) * ed[k, 0]
            gy += (t[i, j, k] - teq) * ed[k, 1]
        denom = -(tauT_field[i, j] * dtT / 3.0)
        Grad_Temp[i, j, 0] = gx / denom
        Grad_Temp[i, j, 1] = gy / denom

@wp.kernel
def macroH_kernel(
    space: wp.array2d(dtype=wp.int32),
    h: wp.array3d(dtype=dtype),
    UC: wp.array2d(dtype=dtype),
    u: wp.array3d(dtype=dtype),
    tauH_field: wp.array2d(dtype=dtype),
    Grad_UC: wp.array3d(dtype=dtype),
    FU: wp.array2d(dtype=dtype),
):
    i, j = wp.tid()
    if space[i, j] == 0:
        uc = 0.0
        for k in range(Q):
            uc += h[i, j, k]
        UC[i, j] = uc
        gx = 0.0
        gy = 0.0
        for k in range(1, Q):
            heq = Heq(k, uc, u[i, j, 0], u[i, j, 1])
            gx += (h[i, j, k] - heq) * ed[k, 0]
            gy += (h[i, j, k] - heq) * ed[k, 1]
        denom = -(tauH_field[i, j] * dtH / 3.0)
        Grad_UC[i, j, 0] = gx / denom
        Grad_UC[i, j, 1] = gy / denom
        FU[i, j] = 1.0 / Le

# ============================================================
# Collision kernels
# ============================================================
@wp.kernel
def mrtF_kernel(
    space: wp.array2d(dtype=wp.int32),
    rho: wp.array2d(dtype=dtype),
    u: wp.array3d(dtype=dtype),
    BF: wp.array3d(dtype=dtype),
    f: wp.array3d(dtype=dtype),
):
    i, j = wp.tid()
    if space[i, j] == 0:
        ux = u[i, j, 0] / cF
        uy = u[i, j, 1] / cF
        uu = ux * ux + uy * uy

        meq = vec9f()
        meq[0] = 1.0
        meq[1] = -2.0 + 3.0 * uu
        meq[2] = 1.0 - 3.0 * uu
        meq[3] = ux
        meq[4] = -ux
        meq[5] = uy
        meq[6] = -uy
        meq[7] = ux * ux - uy * uy
        meq[8] = ux * uy

        Fx = BF[i, j, 0]
        Fy = BF[i, j, 1]
        temp = 6.0 * (Fx * ux + Fy * uy)
        Guo = vec9f()
        Guo[0] = 0.0
        Guo[1] = temp
        Guo[2] = -temp
        Guo[3] = Fx
        Guo[4] = -Fx
        Guo[5] = Fy
        Guo[6] = -Fy
        Guo[7] = 2.0 * (Fx * ux - Fy * uy)
        Guo[8] = Fy * ux + Fx * uy

        tauF = 3.0 * nF * niuL + 0.5
        s1 = 1.19
        s2 = 1.4
        sq = 1.2
        Sf = vec9f(1.0, s1, s2, 1.0, sq, 1.0, sq, 1.0 / tauF, 1.0 / tauF)

        mneq = vec9f()
        force = vec9f()
        rr = rho[i, j]
        for m1 in range(Q):
            mtemp = 0.0
            for n1 in range(Q):
                mtemp += M[m1, n1] * f[i, j, n1]
            mneq[m1] = (rr * meq[m1] - mtemp) * Sf[m1]
            force[m1] = Guo[m1] * (1.0 - 0.5 * Sf[m1])

        for m2 in range(Q):
            dm = 0.0
            df = 0.0
            for n2 in range(Q):
                dm += invM[m2, n2] / 36.0 * mneq[n2]
                df += invM[m2, n2] / 36.0 * force[n2]
            f[i, j, m2] += dm + df * dtF

@wp.kernel
def mrtG_kernel(
    space: wp.array2d(dtype=wp.int32),
    psi: wp.array2d(dtype=dtype),
    UC: wp.array2d(dtype=dtype),
    FU: wp.array2d(dtype=dtype),
    Temp: wp.array2d(dtype=dtype),
    Grad_psi: wp.array3d(dtype=dtype),
    as_field: wp.array2d(dtype=dtype),
    v: wp.array3d(dtype=dtype),
    tauG_field: wp.array2d(dtype=dtype),
    g: wp.array3d(dtype=dtype),
    G: wp.array3d(dtype=dtype),
):
    i, j = wp.tid()
    if space[i, j] == 0:
        phi = psi[i, j]
        uc = UC[i, j]
        fu = FU[i, j]
        gpx = Grad_psi[i, j, 0]
        gpy = Grad_psi[i, j, 1]
        gx2 = gpx * gpx
        gy2 = gpy * gpy
        gmode = wp.sqrt(gx2 + gy2)
        mode4 = gmode * gmode * gmode * gmode

        asn = 1.0
        NGx = 0.0
        NGy = 0.0
        if gmode > 1.0e-6:
            asn = 1.0 - 3.0 * eps + 4.0 * eps * (gx2 * gx2 + gy2 * gy2) / mode4
            NGx = 16.0 * eps * asn * gpx * gy2 * (gx2 - gy2) / mode4
            NGy = 16.0 * eps * asn * gpy * gx2 * (gy2 - gx2) / mode4

        as_field[i, j] = asn
        v[i, j, 0] = -NGx * W0 * W0 / (fu * tau0)
        v[i, j, 1] = -NGy * W0 * W0 / (fu * tau0)

        theta = CPL * (Temp[i, j] - Tm - mm * Ceq) / Latent
        QG = (phi - lamda * (theta + MC * uc) * (1.0 - phi * phi)) * (1.0 - phi * phi) / (fu * tau0)
        vx = v[i, j, 0] / cG
        vy = v[i, j, 1] / cG

        mgeq = vec9f()
        mgeq[0] = phi
        mgeq[1] = -2.0 * phi
        mgeq[2] = phi
        mgeq[3] = vx
        mgeq[4] = -vx
        mgeq[5] = vy
        mgeq[6] = -vy

        tauG = 3.0 * nG * (asn * asn * W0 * W0 / (fu * tau0)) + 0.5
        tauG_field[i, j] = tauG
        Sg = vec9f(1.0, 1.0, 1.0, 1.0 / tauG, 1.0, 1.0 / tauG, 1.0, 1.0, 1.0)

        mneq = vec9f()
        for m1 in range(Q):
            mtemp = 0.0
            for n1 in range(Q):
                mtemp += M[m1, n1] * g[i, j, n1]
            mneq[m1] = (mgeq[m1] - mtemp) * Sg[m1]

        for m2 in range(Q):
            G[i, j, m2] = g[i, j, m2]
            dm = 0.0
            for n2 in range(Q):
                dm += invM[m2, n2] / 36.0 * mneq[n2]
            g[i, j, m2] += dm + w[m2] * QG * dtG

@wp.kernel
def mrtT_kernel(
    space: wp.array2d(dtype=wp.int32),
    fs: wp.array2d(dtype=dtype),
    Temp: wp.array2d(dtype=dtype),
    u: wp.array3d(dtype=dtype),
    Grad_psi: wp.array3d(dtype=dtype),
    Grad_Temp: wp.array3d(dtype=dtype),
    psiAccumT: wp.array2d(dtype=dtype),
    tauT_field: wp.array2d(dtype=dtype),
    t: wp.array3d(dtype=dtype),
):
    i, j = wp.tid()
    if space[i, j] == 0:
        FS = fs[i, j]
        TT0 = Temp[i, j]
        CPeff = CPS * FS + CPL * (1.0 - FS)
        Xeff = (kS * FS + kL * (1.0 - FS)) / CPeff
        ux = u[i, j, 0] / cT
        uy = u[i, j, 1] / cT
        gpx = Grad_psi[i, j, 0]
        gpy = Grad_psi[i, j, 1]
        gtx = Grad_Temp[i, j, 0]
        gty = Grad_Temp[i, j, 1]
        QT_cp = 0.5 * Xeff * (CPS - CPL) / CPeff * (gpx * gtx + gpy * gty)
        QT_latent = Latent * psiAccumT[i, j] / (2.0 * CPeff * dtT)
        psiAccumT[i, j] = 0.0

        mteq = vec9f()
        mteq[0] = TT0
        mteq[1] = -2.0 * TT0
        mteq[2] = TT0
        mteq[3] = ux * TT0
        mteq[4] = -ux * TT0
        mteq[5] = uy * TT0
        mteq[6] = -uy * TT0

        tauT = 3.0 * nT * Xeff + 0.5
        tauT_field[i, j] = tauT
        St = vec9f(1.0, 1.0, 1.0, 1.0 / tauT, 1.0, 1.0 / tauT, 1.0, 1.0, 1.0)

        mneq = vec9f()
        for m1 in range(Q):
            mtemp = 0.0
            for n1 in range(Q):
                mtemp += M[m1, n1] * t[i, j, n1]
            mneq[m1] = (mteq[m1] - mtemp) * St[m1]

        src = QT_cp + QT_latent
        for m2 in range(Q):
            dm = 0.0
            for n2 in range(Q):
                dm += invM[m2, n2] / 36.0 * mneq[n2]
            t[i, j, m2] += dm + w[m2] * src * dtT

@wp.kernel
def mrtH_kernel(
    space: wp.array2d(dtype=wp.int32),
    UC: wp.array2d(dtype=dtype),
    psi: wp.array2d(dtype=dtype),
    u: wp.array3d(dtype=dtype),
    Grad_psi: wp.array3d(dtype=dtype),
    Grad_UC: wp.array3d(dtype=dtype),
    psiAccumH: wp.array2d(dtype=dtype),
    JatDivAccumH: wp.array2d(dtype=dtype),
    tauH_field: wp.array2d(dtype=dtype),
    C: wp.array2d(dtype=dtype),
    h: wp.array3d(dtype=dtype),
):
    i, j = wp.tid()
    if space[i, j] == 0:
        uc = UC[i, j]
        phi = psi[i, j]
        ux = u[i, j, 0] / cH
        uy = u[i, j, 1] / cH
        gpx = Grad_psi[i, j, 0]
        gpy = Grad_psi[i, j, 1]
        gux = Grad_UC[i, j, 0]
        guy = Grad_UC[i, j, 1]
        Dpsi_DU = gpx * gux + gpy * guy
        dt_psi_avg = psiAccumH[i, j] / dtH
        DJat_avg = JatDivAccumH[i, j] / dtH
        psiAccumH[i, j] = 0.0
        JatDivAccumH[i, j] = 0.0
        k_phi = (1.0 + k0) - (1.0 - k0) * phi
        k_U = 1.0 + (1.0 - k0) * uc
        DLS_phi = (1.0 + phi) * DS + (1.0 - phi) * DL
        Deff = DLS_phi / k_phi
        C[i, j] = 0.5 * Ceq * k_phi * k_U

        mheq = vec9f()
        mheq[0] = uc
        mheq[1] = -2.0 * uc
        mheq[2] = uc
        mheq[3] = ux * uc
        mheq[4] = -ux * uc
        mheq[5] = uy * uc
        mheq[6] = -uy * uc

        temp1 = (1.0 - k0) * DLS_phi / (k_phi * k_phi) * Dpsi_DU
        temp2 = (k_U * dt_psi_avg - 2.0 * DJat_avg) / k_phi
        QH = -temp1 + temp2

        tauH = 3.0 * nH * Deff + 0.5
        tauH_field[i, j] = tauH
        s3 = 1.0 / tauH
        s1 = 1.0
        s4 = 2.0 - s1
        Sh = vec9f(1.0, s1, 1.0, s3, s4, s3, s4, s1, s1)

        mneq = vec9f()
        for m1 in range(Q):
            mtemp = 0.0
            for n1 in range(Q):
                mtemp += M[m1, n1] * h[i, j, n1]
            mneq[m1] = (mheq[m1] - mtemp) * Sh[m1]

        for m2 in range(Q):
            dm = 0.0
            for n2 in range(Q):
                dm += invM[m2, n2] / 36.0 * mneq[n2]
            h[i, j, m2] += dm + w[m2] * QH * dtH

# ============================================================
# Source accumulation kernels
# ============================================================
@wp.kernel
def accumulate_sources_stage1(
    space: wp.array2d(dtype=wp.int32),
    psi: wp.array2d(dtype=dtype),
    psi0: wp.array2d(dtype=dtype),
    psiAccumT: wp.array2d(dtype=dtype),
    psiAccumH: wp.array2d(dtype=dtype),
    Grad_psi: wp.array3d(dtype=dtype),
    UC: wp.array2d(dtype=dtype),
    Jat: wp.array3d(dtype=dtype),
):
    i, j = wp.tid()
    if space[i, j] == 0:
        dpsi = psi[i, j] - psi0[i, j]
        psiAccumT[i, j] += dpsi
        psiAccumH[i, j] += dpsi
        gpx = Grad_psi[i, j, 0]
        gpy = Grad_psi[i, j, 1]
        gmode = wp.sqrt(gpx * gpx + gpy * gpy)
        temp = dpsi * W0 / (2.0 * wp.sqrt(2.0)) * (1.0 + (1.0 - k0) * UC[i, j])
        if gmode > 1.0e-6:
            Jat[0, i, j] = -temp * gpx / gmode
            Jat[1, i, j] = -temp * gpy / gmode
        else:
            Jat[0, i, j] = 0.0
            Jat[1, i, j] = 0.0

@wp.kernel
def accumulate_sources_stage2(
    space: wp.array2d(dtype=wp.int32),
    Jat0: wp.array2d(dtype=dtype),
    Jat1: wp.array2d(dtype=dtype),
    JatDivAccumH: wp.array2d(dtype=dtype),
):
    i, j = wp.tid()
    if space[i, j] == 0:
        d0 = grad_scalar(Jat0, i, j, 0)
        d1 = grad_scalar(Jat1, i, j, 1)
        JatDivAccumH[i, j] += d0 + d1

# ============================================================
# Streaming kernels
# ============================================================
@wp.kernel
def streamingF_kernel(
    space: wp.array2d(dtype=wp.int32),
    fs: wp.array2d(dtype=dtype),
    f: wp.array3d(dtype=dtype),
    F: wp.array3d(dtype=dtype),
):
    i, j = wp.tid()
    if space[i, j] == 0:
        for k in range(Q):
            ip = (i + NX + ei[k, 0]) % NX
            jp = (j + NY + ei[k, 1]) % NY
            if space[ip, jp] == 1:
                F[i, j, invk[k]] = f[i, j, k]
            else:
                fss = 0.5 * (fs[i, j] + fs[ip, jp])
                F[ip, jp, k] = fss * f[ip, jp, invk[k]] + (1.0 - fss) * f[i, j, k]

@wp.kernel
def streamingG_kernel(
    space: wp.array2d(dtype=wp.int32),
    as_field: wp.array2d(dtype=dtype),
    g: wp.array3d(dtype=dtype),
    G: wp.array3d(dtype=dtype),
):
    i, j = wp.tid()
    if space[i, j] == 0:
        for k in range(Q):
            ip = (i + NX + ei[k, 0]) % NX
            jp = (j + NY + ei[k, 1]) % NY
            if space[ip, jp] == 1:
                G[i, j, invk[k]] = g[i, j, k]
            else:
                asn = as_field[i, j]
                G[ip, jp, k] = (g[i, j, k] - (1.0 - asn * asn) * G[ip, jp, k]) / (asn * asn)

@wp.kernel
def streaming_passive_kernel(
    space: wp.array2d(dtype=wp.int32),
    src: wp.array3d(dtype=dtype),
    dst: wp.array3d(dtype=dtype),
):
    i, j = wp.tid()
    if space[i, j] == 0:
        for k in range(Q):
            ip = (i + NX + ei[k, 0]) % NX
            jp = (j + NY + ei[k, 1]) % NY
            if space[ip, jp] == 1:
                dst[i, j, invk[k]] = src[i, j, k]
            else:
                dst[ip, jp, k] = src[i, j, k]

# ============================================================
# Boundary kernels
# ============================================================
@wp.kernel
def boundaryF_y_kernel(
    rho: wp.array2d(dtype=dtype),
    u: wp.array3d(dtype=dtype),
    f: wp.array3d(dtype=dtype),
):
    j = wp.tid()
    for k in range(Q):
        rho[0, j] = rho[1, j]
        u[0, j, 0] = INLET_UX
        u[0, j, 1] = u[1, j, 1]
        if ei[k, 0] == 1:
            f[0, j, k] = f[1, j, k] + Feq(k, rho[0, j], u[0, j, 0], u[0, j, 1]) - Feq(k, rho[1, j], u[1, j, 0], u[1, j, 1])

        rho[NX - 1, j] = rho[NX - 2, j]
        u[NX - 1, j, 0] = u[NX - 2, j, 0]
        u[NX - 1, j, 1] = u[NX - 2, j, 1]
        if ei[k, 0] == -1:
            f[NX - 1, j, k] = f[NX - 2, j, k] + Feq(k, rho[NX - 1, j], u[NX - 1, j, 0], u[NX - 1, j, 1]) - Feq(k, rho[NX - 2, j], u[NX - 2, j, 0], u[NX - 2, j, 1])

@wp.kernel
def boundaryG_x_kernel(
    psi: wp.array2d(dtype=dtype),
    v: wp.array3d(dtype=dtype),
    g: wp.array3d(dtype=dtype),
):
    i = wp.tid()
    for k in range(Q):
        psi[i, 0] = psi[i, 1]
        psi[i, NY - 1] = psi[i, NY - 2]
        g[i, 0, k] = g[i, 1, k] + Geq(k, psi[i, 0], v[i, 0, 0], v[i, 0, 1]) - Geq(k, psi[i, 1], v[i, 1, 0], v[i, 1, 1])
        g[i, NY - 1, k] = g[i, NY - 2, k] + Geq(k, psi[i, NY - 1], v[i, NY - 1, 0], v[i, NY - 1, 1]) - Geq(k, psi[i, NY - 2], v[i, NY - 2, 0], v[i, NY - 2, 1])

@wp.kernel
def boundaryG_y_kernel(
    psi: wp.array2d(dtype=dtype),
    v: wp.array3d(dtype=dtype),
    g: wp.array3d(dtype=dtype),
):
    j = wp.tid()
    for k in range(Q):
        psi[0, j] = psi[1, j]
        psi[NX - 1, j] = psi[NX - 2, j]
        g[0, j, k] = g[1, j, k] + Geq(k, psi[0, j], v[0, j, 0], v[0, j, 1]) - Geq(k, psi[1, j], v[1, j, 0], v[1, j, 1])
        g[NX - 1, j, k] = g[NX - 2, j, k] + Geq(k, psi[NX - 1, j], v[NX - 1, j, 0], v[NX - 1, j, 1]) - Geq(k, psi[NX - 2, j], v[NX - 2, j, 0], v[NX - 2, j, 1])

@wp.kernel
def boundaryT_x_kernel(
    Temp: wp.array2d(dtype=dtype),
    u: wp.array3d(dtype=dtype),
    t: wp.array3d(dtype=dtype),
):
    i = wp.tid()
    for k in range(Q):
        Temp[i, 0] = Temp[i, 1]
        Temp[i, NY - 1] = Temp[i, NY - 2]
        t[i, 0, k] = t[i, 1, k] + Teq(k, Temp[i, 0], u[i, 0, 0], u[i, 0, 1]) - Teq(k, Temp[i, 1], u[i, 1, 0], u[i, 1, 1])
        t[i, NY - 1, k] = t[i, NY - 2, k] + Teq(k, Temp[i, NY - 1], u[i, NY - 1, 0], u[i, NY - 1, 1]) - Teq(k, Temp[i, NY - 2], u[i, NY - 2, 0], u[i, NY - 2, 1])

@wp.kernel
def boundaryT_y_kernel(
    Temp: wp.array2d(dtype=dtype),
    u: wp.array3d(dtype=dtype),
    t: wp.array3d(dtype=dtype),
):
    j = wp.tid()
    for k in range(Q):
        Temp[0, j] = Temp[1, j]
        Temp[NX - 1, j] = Temp[NX - 2, j]
        t[0, j, k] = t[1, j, k] + Teq(k, Temp[0, j], u[0, j, 0], u[0, j, 1]) - Teq(k, Temp[1, j], u[1, j, 0], u[1, j, 1])
        t[NX - 1, j, k] = t[NX - 2, j, k] + Teq(k, Temp[NX - 1, j], u[NX - 1, j, 0], u[NX - 1, j, 1]) - Teq(k, Temp[NX - 2, j], u[NX - 2, j, 0], u[NX - 2, j, 1])

@wp.kernel
def boundaryH_x_kernel(
    UC: wp.array2d(dtype=dtype),
    u: wp.array3d(dtype=dtype),
    h: wp.array3d(dtype=dtype),
):
    i = wp.tid()
    for k in range(Q):
        UC[i, 0] = UC[i, 1]
        UC[i, NY - 1] = UC[i, NY - 2]
        h[i, 0, k] = h[i, 1, k] + Heq(k, UC[i, 0], u[i, 0, 0], u[i, 0, 1]) - Heq(k, UC[i, 1], u[i, 1, 0], u[i, 1, 1])
        h[i, NY - 1, k] = h[i, NY - 2, k] + Heq(k, UC[i, NY - 1], u[i, NY - 1, 0], u[i, NY - 1, 1]) - Heq(k, UC[i, NY - 2], u[i, NY - 2, 0], u[i, NY - 2, 1])

@wp.kernel
def boundaryH_y_kernel(
    UC: wp.array2d(dtype=dtype),
    u: wp.array3d(dtype=dtype),
    h: wp.array3d(dtype=dtype),
):
    j = wp.tid()
    for k in range(Q):
        UC[0, j] = UC[1, j]
        UC[NX - 1, j] = UC[NX - 2, j]
        h[0, j, k] = h[1, j, k] + Heq(k, UC[0, j], u[0, j, 0], u[0, j, 1]) - Heq(k, UC[1, j], u[1, j, 0], u[1, j, 1])
        h[NX - 1, j, k] = h[NX - 2, j, k] + Heq(k, UC[NX - 1, j], u[NX - 1, j, 0], u[NX - 1, j, 1]) - Heq(k, UC[NX - 2, j], u[NX - 2, j, 0], u[NX - 2, j, 1])


def read_fields():
    wp.synchronize()
    return {"psi": psi.numpy(), "U": UC.numpy(), "theta": CPL * (Temp.numpy() - Tm - mm * Ceq) / Latent, "velocity": u.numpy()}


def main():
    global f, F, g, G, t, T, h, H
    session = Session(CFG, globals())
    try:
        wp.launch(initialize_fields, dim=dim_space, inputs=[space, rho, u, v, UC, Temp, psi, psi0, fs, fl, FU])
        wp.launch(initialize_tau_and_eq, dim=dim_space, inputs=[space, rho, u, v, UC, Temp, psi, fs, FU, as_field, tauG_field, tauH_field, tauT_field, f, g, t, h])

        if FLOW:
            wp.launch(macroF_kernel, dim=dim_space, inputs=[space, f, rho, BF, u])
        wp.launch(macroG_kernel, dim=dim_space, inputs=[space, g, psi, psi0, v, tauG_field, Grad_psi, fs, fl])
        wp.launch(macroH_kernel, dim=dim_space, inputs=[space, h, UC, u, tauH_field, Grad_UC, FU])
        wp.synchronize()
        session.observe(0, read_fields)
        for step in range(1, CFG.steps + 1):
            if FLOW and step % mF == 0:
                wp.launch(mrtF_kernel, dim=dim_space, inputs=[space, rho, u, BF, f])
                wp.launch(streamingF_kernel, dim=dim_space, inputs=[space, fs, f, F])
                f, F = F, f
                wp.launch(boundaryF_y_kernel, dim=NY, inputs=[rho, u, f])
                wp.launch(macroF_kernel, dim=dim_space, inputs=[space, f, rho, BF, u])

            if step % mG == 0:
                wp.launch(mrtG_kernel, dim=dim_space, inputs=[space, psi, UC, FU, Temp, Grad_psi, as_field, v, tauG_field, g, G])
                wp.launch(streamingG_kernel, dim=dim_space, inputs=[space, as_field, g, G])
                g, G = G, g
                wp.launch(boundaryG_x_kernel, dim=NX, inputs=[psi, v, g])
                wp.launch(boundaryG_y_kernel, dim=NY, inputs=[psi, v, g])
                wp.launch(macroG_kernel, dim=dim_space, inputs=[space, g, psi, psi0, v, tauG_field, Grad_psi, fs, fl])
                wp.launch(accumulate_sources_stage1, dim=dim_space, inputs=[space, psi, psi0, psiAccumT, psiAccumH, Grad_psi, UC, Jat])
                wp.launch(accumulate_sources_stage2, dim=dim_space, inputs=[space, Jat[0], Jat[1], JatDivAccumH])


            if step % mH == 0:
                wp.launch(mrtH_kernel, dim=dim_space, inputs=[space, UC, psi, u, Grad_psi, Grad_UC, psiAccumH, JatDivAccumH, tauH_field, C, h])
                wp.launch(streaming_passive_kernel, dim=dim_space, inputs=[space, h, H])
                h, H = H, h
                wp.launch(boundaryH_x_kernel, dim=NX, inputs=[UC, u, h])
                wp.launch(boundaryH_y_kernel, dim=NY, inputs=[UC, u, h])
                wp.launch(macroH_kernel, dim=dim_space, inputs=[space, h, UC, u, tauH_field, Grad_UC, FU])

            session.observe(step, read_fields)
        session.finish(read_fields)
    except BaseException as exc:
        session.fail(exc)
        raise


if __name__ == "__main__":
    main()
