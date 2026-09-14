"""Thermal dendritic solidification, manuscript section 4.1.

Run `python thermal.py --help`. Numerical provenance and deliberate changes:
docs/MANUSCRIPT_ALIGNMENT.md. Each invocation is an isolated simulation process.
"""
import math
from runtime import configure, Session

CFG = configure("thermal")
import warp as wp
wp.init()
if CFG.device.startswith("cuda") and not wp.is_cuda_available():
    raise RuntimeError("CUDA requested but unavailable; check the NVIDIA driver or use --device cpu")
wp.set_device(CFG.device)
DEVICE = str(wp.get_device())
FLOW = CFG.flow == "on"

dtype = wp.float32

NX = CFG.nx
NY = CFG.ny
Q = 9

TIP_PSI_THRESHOLD = 0.0
TIP_CENTER_X = NX // 2
TIP_CENTER_Y = NY // 2



# ============================================================
# D2Q9 lattice
# ============================================================
dim_space = (NX, NY)
dim_vec2 = (NX, NY, 2)
dim_f = (NX, NY, Q)

vec9f = wp.types.vector(length=Q, dtype=dtype)
vec9i = wp.types.vector(length=Q, dtype=wp.int32)
mat92f = wp.types.matrix(shape=(Q, 2), dtype=dtype)
mat99f = wp.types.matrix(shape=(Q, Q), dtype=dtype)

w = wp.constant(vec9f(
    4.0 / 9.0,
    1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0,
    1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0,
))

ei = wp.constant(wp.types.matrix(shape=(Q, 2), dtype=wp.int32)([
    [0, 0], [1, 0], [0, 1], [-1, 0], [0, -1],
    [1, 1], [-1, 1], [-1, -1], [1, -1],
]))

ed = wp.constant(mat92f([
    [0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0],
    [1.0, 1.0], [-1.0, 1.0], [-1.0, -1.0], [1.0, -1.0],
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

mF = int(round(nF / nF))
mG = int(round(nG / nF))
mT = int(round(nT / nF))

dtF = nF * dx * dx
dtG = nG * dx * dx
dtT = nT * dx * dx
cF = dx / dtF
cG = dx / dtG
cT = dx / dtT

Pe = 0.25
Pr = 23.1
Le = 1.0
W0 = 2.5 * dx
tau0 = 125.0 * dtG
eps_aniso = 0.05

a1 = 5.0 * math.sqrt(2.0) / 8.0
a2 = 47.0 / 75.0
alpha = W0 * W0 / (tau0 * Pe)
lamda = tau0 * alpha / (a2 * W0 * W0)
d0 = a1 * W0 / lamda
niuL = Pr * alpha

MC = 0.0
U0 = 0.0
T0 = -0.55
FU0 = 1.0 / Le
SEED_RADIUS = 10.0
INLET_UX = (W0 / tau0) if FLOW else 0.0

# ============================================================
# Device arrays
# ============================================================
f = wp.zeros(dim_f, dtype=dtype, device=DEVICE)
F = wp.zeros(dim_f, dtype=dtype, device=DEVICE)
g = wp.zeros(dim_f, dtype=dtype, device=DEVICE)
G = wp.zeros(dim_f, dtype=dtype, device=DEVICE)
t_dist = wp.zeros(dim_f, dtype=dtype, device=DEVICE)
T_dist = wp.zeros(dim_f, dtype=dtype, device=DEVICE)

rho = wp.zeros(dim_space, dtype=dtype, device=DEVICE)
psi = wp.zeros(dim_space, dtype=dtype, device=DEVICE)
psi0 = wp.zeros(dim_space, dtype=dtype, device=DEVICE)
theta = wp.zeros(dim_space, dtype=dtype, device=DEVICE)
fs = wp.zeros(dim_space, dtype=dtype, device=DEVICE)
as_field = wp.zeros(dim_space, dtype=dtype, device=DEVICE)
tauG_field = wp.zeros(dim_space, dtype=dtype, device=DEVICE)
psiAccumT = wp.zeros(dim_space, dtype=dtype, device=DEVICE)

u = wp.zeros(dim_vec2, dtype=dtype, device=DEVICE)
v = wp.zeros(dim_vec2, dtype=dtype, device=DEVICE)
BF = wp.zeros(dim_vec2, dtype=dtype, device=DEVICE)
Grad_psi = wp.zeros(dim_vec2, dtype=dtype, device=DEVICE)


# ============================================================
# Warp functions
# ============================================================
SEED_WIDTH = math.sqrt(2.0) * W0 if CFG.parameters == "paper" else math.sqrt(2.0 * W0)

@wp.func
def clamp_i(value: int, lower: int, upper: int):
    if value < lower:
        return lower
    if value > upper:
        return upper
    return value


@wp.func
def grad_scalar(arr: wp.array2d(dtype=dtype), i: int, j: int, xy: int):
    total = float(0.0)
    for k in range(1, Q):
        ip = clamp_i(i + ei[k, 0], 0, NX - 1)
        jp = clamp_i(j + ei[k, 1], 0, NY - 1)
        total += w[k] * arr[ip, jp] * ed[k, xy]
    return 3.0 * total


@wp.func
def Feq(k: int, density: float, ux: float, uy: float):
    eu = (ed[k, 0] * ux + ed[k, 1] * uy) / cF
    uv = (ux * ux + uy * uy) / (cF * cF)
    return w[k] * density * (1.0 + 3.0 * eu + 4.5 * eu * eu - 1.5 * uv)


@wp.func
def Geq(k: int, phi: float, vx: float, vy: float):
    ev = (ed[k, 0] * vx + ed[k, 1] * vy) / cG
    return w[k] * (phi + 3.0 * ev)


@wp.func
def Teq(k: int, temp: float, ux: float, uy: float):
    eu = (ed[k, 0] * ux + ed[k, 1] * uy) / cT
    return w[k] * temp * (1.0 + 3.0 * eu)

# ============================================================
# Initialization
# ============================================================
@wp.kernel
def initialize_fields_kernel(
    rho: wp.array2d(dtype=dtype),
    u: wp.array3d(dtype=dtype),
    v: wp.array3d(dtype=dtype),
    psi: wp.array2d(dtype=dtype),
    psi0: wp.array2d(dtype=dtype),
    theta: wp.array2d(dtype=dtype),
    fs: wp.array2d(dtype=dtype),
):
    i, j = wp.tid()
    rho[i, j] = 1.0
    u[i, j, 0] = 0.0
    u[i, j, 1] = 0.0
    v[i, j, 0] = 0.0
    v[i, j, 1] = 0.0
    theta[i, j] = T0

    x = float(i) - float(NX // 2)
    y = float(j) - float(NY // 2)
    radius = wp.sqrt(x * x + y * y)
    phi = wp.tanh((SEED_RADIUS - radius) / SEED_WIDTH)
    psi[i, j] = phi
    psi0[i, j] = phi
    fs[i, j] = 0.5 * (phi + 1.0)


@wp.kernel
def initialize_distributions_kernel(
    rho: wp.array2d(dtype=dtype),
    u: wp.array3d(dtype=dtype),
    v: wp.array3d(dtype=dtype),
    psi: wp.array2d(dtype=dtype),
    theta: wp.array2d(dtype=dtype),
    as_field: wp.array2d(dtype=dtype),
    tauG_field: wp.array2d(dtype=dtype),
    f: wp.array3d(dtype=dtype),
    g: wp.array3d(dtype=dtype),
    t_dist: wp.array3d(dtype=dtype),
):
    i, j = wp.tid()
    phi = psi[i, j]
    gpx = grad_scalar(psi, i, j, 0)
    gpy = grad_scalar(psi, i, j, 1)
    gx2 = gpx * gpx
    gy2 = gpy * gpy
    gmode = wp.sqrt(gx2 + gy2)

    asn = 1.0
    ngx = 0.0
    ngy = 0.0
    if gmode > 1.0e-6:
        mode4 = gmode * gmode * gmode * gmode
        asn = 1.0 - 3.0 * eps_aniso + 4.0 * eps_aniso * (gx2 * gx2 + gy2 * gy2) / mode4
        ngx = 16.0 * eps_aniso * asn * gpx * gy2 * (gx2 - gy2) / mode4
        ngy = 16.0 * eps_aniso * asn * gpy * gx2 * (gy2 - gx2) / mode4

    vx = -ngx * W0 * W0 / (FU0 * tau0)
    vy = -ngy * W0 * W0 / (FU0 * tau0)
    v[i, j, 0] = vx
    v[i, j, 1] = vy
    as_field[i, j] = asn
    tauG_field[i, j] = 3.0 * nG * (asn * asn * W0 * W0 / (FU0 * tau0)) + 0.5

    for k in range(Q):
        f[i, j, k] = Feq(k, rho[i, j], u[i, j, 0], u[i, j, 1])
        g[i, j, k] = Geq(k, phi, vx, vy)
        t_dist[i, j, k] = Teq(k, theta[i, j], u[i, j, 0], u[i, j, 1])

# ============================================================
# Macroscopic reconstruction
# ============================================================
@wp.kernel
def macro_f_kernel(
    f: wp.array3d(dtype=dtype),
    rho: wp.array2d(dtype=dtype),
    BF: wp.array3d(dtype=dtype),
    u: wp.array3d(dtype=dtype),
):
    i, j = wp.tid()
    density = 0.0
    momentum_x = 0.0
    momentum_y = 0.0
    for k in range(Q):
        value = f[i, j, k]
        density += value
        momentum_x += value * ed[k, 0]
        momentum_y += value * ed[k, 1]
    rho[i, j] = density
    u[i, j, 0] = (momentum_x + 0.5 * dtF * BF[i, j, 0]) / density * cF
    u[i, j, 1] = (momentum_y + 0.5 * dtF * BF[i, j, 1]) / density * cF


@wp.kernel
def macro_g_kernel(
    g: wp.array3d(dtype=dtype),
    psi: wp.array2d(dtype=dtype),
    psi0: wp.array2d(dtype=dtype),
    v: wp.array3d(dtype=dtype),
    tauG_field: wp.array2d(dtype=dtype),
    Grad_psi: wp.array3d(dtype=dtype),
    fs: wp.array2d(dtype=dtype),
):
    i, j = wp.tid()
    old_phi = psi[i, j]
    phi = 0.0
    for k in range(Q):
        phi += g[i, j, k]
    psi0[i, j] = old_phi
    psi[i, j] = phi

    gpx = 0.0
    gpy = 0.0
    for k in range(1, Q):
        geq = Geq(k, phi, v[i, j, 0], v[i, j, 1])
        gpx += (g[i, j, k] - geq) * ed[k, 0]
        gpy += (g[i, j, k] - geq) * ed[k, 1]
    denominator = -(tauG_field[i, j] * dtG / 3.0)
    Grad_psi[i, j, 0] = gpx / denominator
    Grad_psi[i, j, 1] = gpy / denominator
    fs[i, j] = 0.5 * (phi + 1.0)


@wp.kernel
def macro_t_kernel(t_dist: wp.array3d(dtype=dtype), theta: wp.array2d(dtype=dtype)):
    i, j = wp.tid()
    temp = 0.0
    for k in range(Q):
        temp += t_dist[i, j, k]
    theta[i, j] = temp

# ============================================================
# MRT collision
# ============================================================
@wp.kernel
def mrt_f_kernel(
    rho: wp.array2d(dtype=dtype),
    u: wp.array3d(dtype=dtype),
    BF: wp.array3d(dtype=dtype),
    f: wp.array3d(dtype=dtype),
):
    i, j = wp.tid()
    ux = u[i, j, 0] / cF
    uy = u[i, j, 1] / cF
    velocity2 = ux * ux + uy * uy

    meq = vec9f()
    meq[0] = 1.0
    meq[1] = -2.0 + 3.0 * velocity2
    meq[2] = 1.0 - 3.0 * velocity2
    meq[3] = ux
    meq[4] = -ux
    meq[5] = uy
    meq[6] = -uy
    meq[7] = ux * ux - uy * uy
    meq[8] = ux * uy

    fx = BF[i, j, 0]
    fy = BF[i, j, 1]
    force_dot_u = 6.0 * (fx * ux + fy * uy)
    guo = vec9f()
    guo[0] = 0.0
    guo[1] = force_dot_u
    guo[2] = -force_dot_u
    guo[3] = fx
    guo[4] = -fx
    guo[5] = fy
    guo[6] = -fy
    guo[7] = 2.0 * (fx * ux - fy * uy)
    guo[8] = fy * ux + fx * uy

    tau_f = 3.0 * nF * niuL + 0.5
    sf = vec9f(1.0, 1.19, 1.4, 1.0, 1.2, 1.0, 1.2, 1.0 / tau_f, 1.0 / tau_f)

    mneq = vec9f()
    force = vec9f()
    density = rho[i, j]
    for m1 in range(Q):
        moment = 0.0
        for n1 in range(Q):
            moment += M[m1, n1] * f[i, j, n1]
        mneq[m1] = (density * meq[m1] - moment) * sf[m1]
        force[m1] = guo[m1] * (1.0 - 0.5 * sf[m1])

    for m2 in range(Q):
        delta = 0.0
        force_delta = 0.0
        for n2 in range(Q):
            delta += invM[m2, n2] / 36.0 * mneq[n2]
            force_delta += invM[m2, n2] / 36.0 * force[n2]
        f[i, j, m2] += delta + force_delta * dtF


@wp.kernel
def mrt_g_kernel(
    psi: wp.array2d(dtype=dtype),
    theta: wp.array2d(dtype=dtype),
    Grad_psi: wp.array3d(dtype=dtype),
    as_field: wp.array2d(dtype=dtype),
    tauG_field: wp.array2d(dtype=dtype),
    v: wp.array3d(dtype=dtype),
    g: wp.array3d(dtype=dtype),
    G: wp.array3d(dtype=dtype),
):
    i, j = wp.tid()
    phi = psi[i, j]
    gpx = Grad_psi[i, j, 0]
    gpy = Grad_psi[i, j, 1]
    gx2 = gpx * gpx
    gy2 = gpy * gpy
    gmode = wp.sqrt(gx2 + gy2)

    asn = 1.0
    ngx = 0.0
    ngy = 0.0
    if gmode > 1.0e-6:
        mode4 = gmode * gmode * gmode * gmode
        asn = 1.0 - 3.0 * eps_aniso + 4.0 * eps_aniso * (gx2 * gx2 + gy2 * gy2) / mode4
        ngx = 16.0 * eps_aniso * asn * gpx * gy2 * (gx2 - gy2) / mode4
        ngy = 16.0 * eps_aniso * asn * gpy * gx2 * (gy2 - gx2) / mode4

    vx_phys = -ngx * W0 * W0 / (FU0 * tau0)
    vy_phys = -ngy * W0 * W0 / (FU0 * tau0)
    as_field[i, j] = asn
    v[i, j, 0] = vx_phys
    v[i, j, 1] = vy_phys

    one_minus_phi2 = 1.0 - phi * phi
    qg = (phi - lamda * (theta[i, j] + MC * U0) * one_minus_phi2) * one_minus_phi2 / (FU0 * tau0)

    mgeq = vec9f()
    mgeq[0] = phi
    mgeq[1] = -2.0 * phi
    mgeq[2] = phi
    mgeq[3] = vx_phys / cG
    mgeq[4] = -vx_phys / cG
    mgeq[5] = vy_phys / cG
    mgeq[6] = -vy_phys / cG
    mgeq[7] = 0.0
    mgeq[8] = 0.0

    tau_g = 3.0 * nG * (asn * asn * W0 * W0 / (FU0 * tau0)) + 0.5
    tauG_field[i, j] = tau_g
    sg = vec9f(1.0, 1.0, 1.0, 1.0 / tau_g, 1.0, 1.0 / tau_g, 1.0, 1.0, 1.0)

    mneq = vec9f()
    for m1 in range(Q):
        moment = 0.0
        for n1 in range(Q):
            moment += M[m1, n1] * g[i, j, n1]
        mneq[m1] = (mgeq[m1] - moment) * sg[m1]

    for m2 in range(Q):
        G[i, j, m2] = g[i, j, m2]
        delta = 0.0
        for n2 in range(Q):
            delta += invM[m2, n2] / 36.0 * mneq[n2]
        g[i, j, m2] += delta + w[m2] * qg * dtG


@wp.kernel
def accumulate_temperature_source_kernel(
    psi: wp.array2d(dtype=dtype),
    psi0: wp.array2d(dtype=dtype),
    psiAccumT: wp.array2d(dtype=dtype),
):
    i, j = wp.tid()
    psiAccumT[i, j] += psi[i, j] - psi0[i, j]


@wp.kernel
def mrt_t_kernel(
    theta: wp.array2d(dtype=dtype),
    u: wp.array3d(dtype=dtype),
    psiAccumT: wp.array2d(dtype=dtype),
    t_dist: wp.array3d(dtype=dtype),
):
    i, j = wp.tid()
    temp = theta[i, j]
    ux = u[i, j, 0] / cT
    uy = u[i, j, 1] / cT
    qt = psiAccumT[i, j] / (2.0 * dtT)
    psiAccumT[i, j] = 0.0

    mteq = vec9f()
    mteq[0] = temp
    mteq[1] = -2.0 * temp
    mteq[2] = temp
    mteq[3] = ux * temp
    mteq[4] = -ux * temp
    mteq[5] = uy * temp
    mteq[6] = -uy * temp
    mteq[7] = 0.0
    mteq[8] = 0.0

    tau_t = 3.0 * nT * alpha + 0.5
    st = vec9f(1.0, 1.0, 1.0, 1.0 / tau_t, 1.0, 1.0 / tau_t, 1.0, 1.0, 1.0)

    mneq = vec9f()
    for m1 in range(Q):
        moment = 0.0
        for n1 in range(Q):
            moment += M[m1, n1] * t_dist[i, j, n1]
        mneq[m1] = (mteq[m1] - moment) * st[m1]

    for m2 in range(Q):
        delta = 0.0
        for n2 in range(Q):
            delta += invM[m2, n2] / 36.0 * mneq[n2]
        t_dist[i, j, m2] += delta + w[m2] * qt * dtT

# ============================================================
# Streaming
# ============================================================
@wp.kernel
def streaming_f_kernel(
    fs: wp.array2d(dtype=dtype),
    f: wp.array3d(dtype=dtype),
    F: wp.array3d(dtype=dtype),
):
    i, j = wp.tid()
    for k in range(Q):
        ip = (i + NX + ei[k, 0]) % NX
        jp = (j + NY + ei[k, 1]) % NY
        solid_fraction = 0.5 * (fs[i, j] + fs[ip, jp])
        F[ip, jp, k] = solid_fraction * f[ip, jp, invk[k]] + (1.0 - solid_fraction) * f[i, j, k]


@wp.kernel
def streaming_g_kernel(
    as_field: wp.array2d(dtype=dtype),
    g: wp.array3d(dtype=dtype),
    G: wp.array3d(dtype=dtype),
):
    i, j = wp.tid()
    as2 = as_field[i, j] * as_field[i, j]
    for k in range(Q):
        ip = (i + NX + ei[k, 0]) % NX
        jp = (j + NY + ei[k, 1]) % NY
        G[ip, jp, k] = (g[i, j, k] - (1.0 - as2) * G[ip, jp, k]) / as2


@wp.kernel
def streaming_t_kernel(
    t_dist: wp.array3d(dtype=dtype),
    T_dist: wp.array3d(dtype=dtype),
):
    i, j = wp.tid()
    for k in range(Q):
        ip = (i + NX + ei[k, 0]) % NX
        jp = (j + NY + ei[k, 1]) % NY
        T_dist[ip, jp, k] = t_dist[i, j, k]

# ============================================================
# Boundary conditions preserved from the active C++ main loop
# ============================================================
@wp.kernel
def boundary_f_x_kernel(
    rho: wp.array2d(dtype=dtype),
    u: wp.array3d(dtype=dtype),
    f: wp.array3d(dtype=dtype),
):
    j = wp.tid()
    rho[0, j] = rho[1, j]
    u[0, j, 0] = INLET_UX
    u[0, j, 1] = u[1, j, 1]

    rho[NX - 1, j] = rho[NX - 2, j]
    u[NX - 1, j, 0] = u[NX - 2, j, 0]
    u[NX - 1, j, 1] = u[NX - 2, j, 1]

    for k in range(Q):
        if ei[k, 0] == 1:
            f[0, j, k] = (
                f[1, j, k]
                + Feq(k, rho[0, j], u[0, j, 0], u[0, j, 1])
                - Feq(k, rho[1, j], u[1, j, 0], u[1, j, 1])
            )
        if ei[k, 0] == -1:
            f[NX - 1, j, k] = (
                f[NX - 2, j, k]
                + Feq(k, rho[NX - 1, j], u[NX - 1, j, 0], u[NX - 1, j, 1])
                - Feq(k, rho[NX - 2, j], u[NX - 2, j, 0], u[NX - 2, j, 1])
            )


@wp.kernel
def boundary_g_y_kernel(
    psi: wp.array2d(dtype=dtype),
    v: wp.array3d(dtype=dtype),
    g: wp.array3d(dtype=dtype),
):
    i = wp.tid()
    psi[i, 0] = psi[i, 1]
    psi[i, NY - 1] = psi[i, NY - 2]
    for k in range(Q):
        g[i, 0, k] = g[i, 1, k] + Geq(k, psi[i, 0], v[i, 0, 0], v[i, 0, 1]) - Geq(k, psi[i, 1], v[i, 1, 0], v[i, 1, 1])
        g[i, NY - 1, k] = g[i, NY - 2, k] + Geq(k, psi[i, NY - 1], v[i, NY - 1, 0], v[i, NY - 1, 1]) - Geq(k, psi[i, NY - 2], v[i, NY - 2, 0], v[i, NY - 2, 1])


@wp.kernel
def boundary_g_x_kernel(
    psi: wp.array2d(dtype=dtype),
    v: wp.array3d(dtype=dtype),
    g: wp.array3d(dtype=dtype),
):
    j = wp.tid()
    psi[0, j] = psi[1, j]
    psi[NX - 1, j] = psi[NX - 2, j]
    for k in range(Q):
        g[0, j, k] = g[1, j, k] + Geq(k, psi[0, j], v[0, j, 0], v[0, j, 1]) - Geq(k, psi[1, j], v[1, j, 0], v[1, j, 1])
        g[NX - 1, j, k] = g[NX - 2, j, k] + Geq(k, psi[NX - 1, j], v[NX - 1, j, 0], v[NX - 1, j, 1]) - Geq(k, psi[NX - 2, j], v[NX - 2, j, 0], v[NX - 2, j, 1])


@wp.kernel
def boundary_t_y_kernel(
    theta: wp.array2d(dtype=dtype),
    u: wp.array3d(dtype=dtype),
    t_dist: wp.array3d(dtype=dtype),
):
    i = wp.tid()
    theta[i, 0] = theta[i, 1]
    theta[i, NY - 1] = theta[i, NY - 2]
    for k in range(Q):
        t_dist[i, 0, k] = t_dist[i, 1, k] + Teq(k, theta[i, 0], u[i, 0, 0], u[i, 0, 1]) - Teq(k, theta[i, 1], u[i, 1, 0], u[i, 1, 1])
        t_dist[i, NY - 1, k] = t_dist[i, NY - 2, k] + Teq(k, theta[i, NY - 1], u[i, NY - 1, 0], u[i, NY - 1, 1]) - Teq(k, theta[i, NY - 2], u[i, NY - 2, 0], u[i, NY - 2, 1])


@wp.kernel
def boundary_t_x_kernel(
    theta: wp.array2d(dtype=dtype),
    u: wp.array3d(dtype=dtype),
    t_dist: wp.array3d(dtype=dtype),
):
    j = wp.tid()
    theta[0, j] = theta[1, j]
    theta[NX - 1, j] = theta[NX - 2, j]
    for k in range(Q):
        t_dist[0, j, k] = t_dist[1, j, k] + Teq(k, theta[0, j], u[0, j, 0], u[0, j, 1]) - Teq(k, theta[1, j], u[1, j, 0], u[1, j, 1])
        t_dist[NX - 1, j, k] = t_dist[NX - 2, j, k] + Teq(k, theta[NX - 1, j], u[NX - 1, j, 0], u[NX - 1, j, 1]) - Teq(k, theta[NX - 2, j], u[NX - 2, j, 0], u[NX - 2, j, 1])


def read_fields():
    wp.synchronize()
    return {"psi": psi.numpy(), "theta": theta.numpy(), "velocity": u.numpy()}


def main():
    global f, F, g, G, t_dist, T_dist
    session = Session(CFG, globals())
    try:
        wp.launch(initialize_fields_kernel, dim=dim_space, inputs=[rho, u, v, psi, psi0, theta, fs], device=DEVICE)
        wp.launch(
            initialize_distributions_kernel,
            dim=dim_space,
            inputs=[rho, u, v, psi, theta, as_field, tauG_field, f, g, t_dist],
            device=DEVICE,
        )
        if FLOW:
            wp.launch(macro_f_kernel, dim=dim_space, inputs=[f, rho, BF, u], device=DEVICE)
        wp.launch(macro_g_kernel, dim=dim_space, inputs=[g, psi, psi0, v, tauG_field, Grad_psi, fs], device=DEVICE)
        wp.launch(macro_t_kernel, dim=dim_space, inputs=[t_dist, theta], device=DEVICE)
        wp.synchronize()
        session.observe(0, read_fields)
        for step in range(1, CFG.steps + 1):
            if FLOW and step % mF == 0:
                wp.launch(mrt_f_kernel, dim=dim_space, inputs=[rho, u, BF, f], device=DEVICE)
                wp.launch(streaming_f_kernel, dim=dim_space, inputs=[fs, f, F], device=DEVICE)
                f, F = F, f
                wp.launch(boundary_f_x_kernel, dim=NY, inputs=[rho, u, f], device=DEVICE)
                wp.launch(macro_f_kernel, dim=dim_space, inputs=[f, rho, BF, u], device=DEVICE)

            if step % mG == 0:
                wp.launch(
                    mrt_g_kernel,
                    dim=dim_space,
                    inputs=[psi, theta, Grad_psi, as_field, tauG_field, v, g, G],
                    device=DEVICE,
                )
                wp.launch(streaming_g_kernel, dim=dim_space, inputs=[as_field, g, G], device=DEVICE)
                g, G = G, g
                wp.launch(boundary_g_y_kernel, dim=NX, inputs=[psi, v, g], device=DEVICE)
                wp.launch(boundary_g_x_kernel, dim=NY, inputs=[psi, v, g], device=DEVICE)
                wp.launch(macro_g_kernel, dim=dim_space, inputs=[g, psi, psi0, v, tauG_field, Grad_psi, fs], device=DEVICE)
                wp.launch(accumulate_temperature_source_kernel, dim=dim_space, inputs=[psi, psi0, psiAccumT], device=DEVICE)

            if step % mT == 0:
                wp.launch(mrt_t_kernel, dim=dim_space, inputs=[theta, u, psiAccumT, t_dist], device=DEVICE)
                wp.launch(streaming_t_kernel, dim=dim_space, inputs=[t_dist, T_dist], device=DEVICE)
                t_dist, T_dist = T_dist, t_dist
                wp.launch(boundary_t_y_kernel, dim=NX, inputs=[theta, u, t_dist], device=DEVICE)
                wp.launch(boundary_t_x_kernel, dim=NY, inputs=[theta, u, t_dist], device=DEVICE)
                wp.launch(macro_t_kernel, dim=dim_space, inputs=[t_dist, theta], device=DEVICE)

            session.observe(step, read_fields)
        session.finish(read_fields)
    except BaseException as exc:
        session.fail(exc)
        raise


if __name__ == "__main__":
    main()
