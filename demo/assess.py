"""Library for the demo notebook contrastive_generation.ipynb: everything the notebook calls.

Sections, in order: common utilities, configs, data, networks, the pull/push constructions, samplers, metrics,
the teacher, the anchor construction, the student trainer, the exports, the animations and the static figures,
then the orchestration used by the notebook. Training runs are launched as subprocesses of this file:

    python assess.py teacher [--preset smoke] [--device cpu]
    python assess.py train --variant mf-on-all --seed 0 --preset smoke --device cpu --steps 3000 ...

Run artifacts go to demo/artifacts/, figures and animations of the post to materials/.
"""
import argparse
import copy
import csv
import hashlib
import itertools
import json
import math
import os
import random
import shutil
import subprocess
import sys
import textwrap
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.legend import Legend
from matplotlib.legend_handler import HandlerBase
from matplotlib.lines import Line2D
from PIL import Image
import av
import numpy as np
import torch
import torch.nn as nn

# Every number in the post was produced on CPU with one thread per process (runs go side by side instead);
# BLAS reductions with more threads change the results in the last digits.
os.environ.setdefault("OMP_NUM_THREADS", "1")
torch.set_num_threads(1)

# ====================================================================================================
# common: paths, seeding, EMA, schedules, logging
# ====================================================================================================
ROOT = Path(__file__).resolve().parent
ART = ROOT / "artifacts"
MATERIALS = ROOT.parent / "materials"

# The dataset is fixed per process by CG_DATA, and training subprocesses inherit it. Datasets other than the
# two moons of the post keep all their artifacts (teacher, runs, exports) under artifacts/<name>/.
DATA = os.environ.get("CG_DATA", "moons")
DIM = {"moons": 2, "checker3d": 3}[DATA]
if DATA != "moons":
    ART = ART / DATA


def pick_device(name="auto"):
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def cpu_randn(shape, seed):
    """Device-independent noise: drawn on CPU with a fixed generator."""
    g = torch.Generator().manual_seed(seed)
    return torch.randn(*shape, generator=g)


class EMA:
    def __init__(self, model, decay):
        self.decay = decay
        self.shadow = copy.deepcopy(model).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)
        self.n = 0

    @torch.no_grad()
    def update(self, model):
        self.n += 1
        d = min(self.decay, (1 + self.n) / (10 + self.n))
        for s, p in zip(self.shadow.parameters(), model.parameters()):
            s.mul_(d).add_(p.detach(), alpha=1 - d)


def cosine_lr(step, total, lr, lr_min, warmup):
    if step < warmup:
        return lr * (step + 1) / warmup
    p = min(1.0, (step - warmup) / max(1, total - warmup))
    return lr_min + 0.5 * (lr - lr_min) * (1 + math.cos(math.pi * p))


def save_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=_jsonable))


def load_json(path):
    return json.loads(Path(path).read_text())


def _jsonable(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    if isinstance(o, tuple):
        return list(o)
    raise TypeError(type(o))


class CSVLog:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.rows = []

    def add(self, **row):
        self.rows.append(row)

    def save(self):
        if not self.rows:
            return
        keys = []
        for r in self.rows:
            keys += [k for k in r if k not in keys]
        with self.path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(self.rows)


def read_csv(path):
    with Path(path).open() as f:
        rows = list(csv.DictReader(f))
    out = []
    for r in rows:
        out.append({k: _num(v) for k, v in r.items()})
    return out


def _num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return v


# ====================================================================================================
# configs: presets and variant names
# ====================================================================================================
@dataclass
class DataCfg:
    noise: float = 0.05
    n_ref: int = 20000


@dataclass
class TeacherCfg:
    width: int = 256
    depth: int = 4
    n_freq: int = 16
    steps: int = 100_000
    batch: int = 4096
    lr: float = 2e-3
    lr_min: float = 1e-5
    warmup: int = 1000
    ema: float = 0.9995
    clip: float = 1.0
    ref_rk4: int = 100
    log_every: int = 500


@dataclass
class StudentCfg:
    steps: int = 30_000
    batch: int = 2048
    lr: float = 1e-4
    lr_min: float = 1e-6
    warmup: int = 500
    ema: float = 0.999
    clip: float = 1.0
    k_roll: int = 2
    n_fm: int = 5
    fake_lr: float = 3e-4
    sim_rk4: int = 32
    frac_diag: float = 0.25
    frac_zero: float = 0.25
    dmd_tmin: float = 0.02
    dmd_tmax: float = 0.98
    eps_cd: float = 1e-2
    hybrid_lambda: float = 0.5
    log_every: int = 100
    quick_every: int = 1000
    snap_count: int = 40
    snap_n: int = 2000
    quick_n: int = 5000
    quick_proj: int = 200
    milestones: tuple = (0, 100, 200, 500, 2000, 10000)


@dataclass
class EvalCfg:
    n: int = 20000
    n_proj: int = 1000
    knn_n: int = 10000
    knn_k: int = 5
    energy_n: int = 10000
    ks: tuple = (1, 2, 4, 8)
    teacher_ks: tuple = (1, 2, 4, 8, 16, 32)
    tri_nt: int = 20
    tri_states: int = 2000
    tri_rk4: int = 200
    coupling_n: int = 10000
    coupling_rk4: int = 100
    comp_triples: int = 60
    mmd_factors: tuple = (0.25, 0.5, 1.0, 2.0)


@dataclass
class ExportCfg:
    v1_n: int = 2000
    v2_particles: int = 400
    v2_grid: int = 25
    v2_times: int = 21
    hero_candidates: int = 1000
    hero_rk4: int = 400
    hero_max_angle: float = 90.0
    hero_min_chord: float = 0.8
    cloud_n: int = 2000
    fit_demo_steps: tuple = (0, 10, 50, 200, 1000)
    dmd_ts: tuple = (0.9, 0.6, 0.3)


@dataclass
class Preset:
    name: str
    data: DataCfg = field(default_factory=DataCfg)
    teacher: TeacherCfg = field(default_factory=TeacherCfg)
    student: StudentCfg = field(default_factory=StudentCfg)
    eval: EvalCfg = field(default_factory=EvalCfg)
    export: ExportCfg = field(default_factory=ExportCfg)

    def to_dict(self):
        return asdict(self)


def get_preset(name):
    p = _base_preset(name)
    if DATA == "checker3d":
        # The sharp cube faces need a better-trained teacher than the moons do.
        p.teacher = replace(p.teacher, steps=max(p.teacher.steps, 20_000), batch=max(p.teacher.batch, 1024))
    return p


def _base_preset(name):
    if name == "full":
        return Preset("full")
    if name == "smoke":
        return Preset(
            "smoke",
            data=DataCfg(n_ref=4000),
            teacher=TeacherCfg(steps=2000, batch=512, warmup=100, log_every=100, ref_rk4=50),
            student=StudentCfg(
                steps=500, batch=256, warmup=20, log_every=20, quick_every=100,
                snap_count=10, snap_n=500, quick_n=1000, quick_proj=100, sim_rk4=8,
                milestones=(0, 50, 100, 200),
            ),
            eval=EvalCfg(
                n=2000, n_proj=200, knn_n=1000, energy_n=2000, tri_nt=5, tri_states=200,
                tri_rk4=50, coupling_n=1000, coupling_rk4=50, comp_triples=20,
            ),
            export=ExportCfg(v1_n=600, v2_particles=150, v2_grid=15, v2_times=11,
                             hero_candidates=200, hero_rk4=200, cloud_n=500,
                             fit_demo_steps=(0, 10, 50, 200)),
        )
    raise ValueError(f"unknown preset {name}")


EST = ("sim", "mf", "dmd", "hybrid", "drift")
POLICY = ("off", "on")
LEVEL = ("r0", "all")
MAIN_GRID = [f"{e}-{p}-{l}" for e in ("sim", "mf", "dmd") for p in POLICY for l in LEVEL]

# Runs whose raw checkpoints at early steps feed the V4/V5 exports.
MILESTONE_RUNS = ("sim-on-all", "mf-on-all", "mf-on-r0", "mf-off-r0", "dmd-on-r0", "dmd-off-r0", "dmd-on-all")

def parse_variant(v):
    est, policy, level = v.split("-")
    assert est in EST and policy in POLICY and level in LEVEL, v
    return est, policy, level


def run_name(variant, tag=""):
    return variant + (f"+{tag}" if tag else "")


def native_sampler(level):
    """Sampler a student is evaluated with: chord chain if all chords are trained,
    otherwise jump-to-zero and renoise."""
    return "chain" if level == "all" else "renoise"


def override(cfg, **kw):
    return replace(cfg, **{k: v for k, v in kw.items() if v is not None})


# ====================================================================================================
# data: two moons
# ====================================================================================================
def moons_raw(n, noise, generator=None, device="cpu"):
    """Classic two moons. Label 1 is the upper moon."""
    phi = torch.rand(n, generator=generator, device=device) * math.pi
    upper = torch.rand(n, generator=generator, device=device) < 0.5
    xu = torch.stack([torch.cos(phi), torch.sin(phi)], 1)
    xl = torch.stack([1 - torch.cos(phi), 0.5 - torch.sin(phi)], 1)
    x = torch.where(upper[:, None], xu, xl)
    x = x + noise * torch.randn(n, 2, generator=generator, device=device)
    return x, upper.long()


def data_stats(noise):
    """Mean and a single isotropic scale (keeps the moon shape), estimated once."""
    path = ART / "data" / "data_stats.json"
    if path.exists():
        s = load_json(path)
        if abs(s["noise"] - noise) < 1e-12:
            return s
    g = torch.Generator().manual_seed(12345)
    x, _ = moons_raw(1_000_000, noise, g)
    mean = x.mean(0)
    scale = x.var(0).mean().sqrt()
    s = dict(noise=noise, mean=mean.tolist(), scale=scale.item())
    save_json(path, s)
    return s


class Moons:
    def __init__(self, noise=0.05, device="cpu"):
        s = data_stats(noise)
        self.noise = noise
        self.device = torch.device(device)
        self.mean = torch.tensor(s["mean"], device=self.device)
        self.scale = s["scale"]

    def standardize(self, x):
        return (x - self.mean.to(x.device)) / self.scale

    def sample(self, n, with_labels=False):
        x, lab = moons_raw(n, self.noise, device=self.device)
        x = self.standardize(x)
        return (x, lab) if with_labels else x

    def fixed(self, n, seed):
        """Device-independent fixed sample (drawn on CPU)."""
        g = torch.Generator().manual_seed(seed)
        x, lab = moons_raw(n, self.noise, g)
        return self.standardize(x).to(self.device), lab.to(self.device)


CHECKER_CELLS = torch.tensor([c for c in itertools.product(range(3), repeat=3) if sum(c) % 2 == 0],
                             dtype=torch.float32)


def checker3d_raw(n, generator=None):
    """3D checkerboard: uniform on the 14 filled unit cubes of a 3x3x3 grid on [-1.5, 1.5]^3, the cells whose
    indices sum to an even number (the corners and the centre are filled). Label: index of the cube."""
    idx = torch.randint(len(CHECKER_CELLS), (n,), generator=generator)
    return CHECKER_CELLS[idx] + torch.rand(n, 3, generator=generator) - 1.5, idx


class Checker3D:
    """Same interface as Moons; one isotropic scale, so the cubes stay cubes."""

    def __init__(self, device="cpu"):
        self.device = torch.device(device)
        x, _ = checker3d_raw(1_000_000, torch.Generator().manual_seed(12345))
        self.mean = x.mean(0).to(self.device)
        self.scale = x.var(0).mean().sqrt().item()

    def standardize(self, x):
        return (x - self.mean.to(x.device)) / self.scale

    def sample(self, n, with_labels=False):
        x, lab = checker3d_raw(n)
        x, lab = self.standardize(x.to(self.device)), lab.to(self.device)
        return (x, lab) if with_labels else x

    def fixed(self, n, seed):
        x, lab = checker3d_raw(n, torch.Generator().manual_seed(seed))
        return self.standardize(x).to(self.device), lab.to(self.device)

    def edges(self):
        """The 12 edges of every filled cube, standardized, as [n_edges, 2, 3] segments for the 3D plots."""
        corners = torch.tensor(list(itertools.product((0.0, 1.0), repeat=3)))
        pairs = [(a, b) for a, b in itertools.combinations(range(8), 2) if (corners[a] - corners[b]).abs().sum() == 1]
        segs = torch.stack([torch.stack([c + corners[a], c + corners[b]]) for c in CHECKER_CELLS for a, b in pairs])
        return self.standardize(segs - 1.5).cpu().numpy()


def make_data(noise, device):
    return Moons(noise, device) if DATA == "moons" else Checker3D(device)


def load_refs(moons, n):
    """ref_A and ref_B: two independent fixed data sets with moon labels."""
    path = ART / "data" / f"refs_{n}.pt"
    if path.exists():
        d = torch.load(path)
    else:
        a, la = moons.fixed(n, seed=1)
        b, lb = moons.fixed(n, seed=2)
        d = dict(A=a.cpu(), A_lab=la.cpu(), B=b.cpu(), B_lab=lb.cpu())
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(d, path)
    return {k: v.to(moons.device) for k, v in d.items()}


# ====================================================================================================
# nets: the teacher / fake velocity network and the student flow-map network
# ====================================================================================================
def as_col(t, x):
    """Broadcast a time given as float, 0-d, [B] or [B, 1] to a [B, 1] column like x."""
    if not torch.is_tensor(t):
        return torch.full((x.shape[0], 1), float(t), device=x.device, dtype=x.dtype)
    t = t.to(device=x.device, dtype=x.dtype)
    if t.dim() == 0:
        return t.reshape(1, 1).expand(x.shape[0], 1)
    if t.dim() == 1:
        return t.unsqueeze(-1)
    return t


class TimeEmbed(nn.Module):
    """[t, sin(w_k t), cos(w_k t)] with w_k geometric between pi and 16 pi."""

    def __init__(self, n_freq=16):
        super().__init__()
        self.register_buffer("freqs", math.pi * 2.0 ** torch.linspace(0, 4, n_freq))
        self.dim = 1 + 2 * n_freq

    def forward(self, t):
        a = t * self.freqs
        return torch.cat([t, torch.sin(a), torch.cos(a)], dim=-1)


def make_mlp(d_in, width, depth, d_out=DIM):
    layers = [nn.Linear(d_in, width), nn.SiLU()]
    for _ in range(depth - 1):
        layers += [nn.Linear(width, width), nn.SiLU()]
    layers += [nn.Linear(width, d_out)]
    return nn.Sequential(*layers)


class VelocityNet(nn.Module):
    """Instantaneous velocity v(x, t): the teacher and the fake (auxiliary) network."""

    def __init__(self, width=256, depth=4, n_freq=16):
        super().__init__()
        self.cfg = dict(width=width, depth=depth, n_freq=n_freq)
        self.emb = TimeEmbed(n_freq)
        self.net = make_mlp(DIM + self.emb.dim, width, depth)

    def forward(self, x, t):
        return self.net(torch.cat([x, self.emb(as_col(t, x))], -1))


class FlowMapNet(nn.Module):
    """Average velocity u(x, t, r): the student. Its jump is x_r = x_t - (t - r) u."""

    def __init__(self, width=256, depth=4, n_freq=16):
        super().__init__()
        self.cfg = dict(width=width, depth=depth, n_freq=n_freq)
        self.emb = TimeEmbed(n_freq)
        self.net = make_mlp(DIM + 2 * self.emb.dim, width, depth)

    def forward(self, x, t, r):
        return self.net(torch.cat([x, self.emb(as_col(t, x)), self.emb(as_col(r, x))], -1))

    @classmethod
    def from_teacher(cls, teacher):
        """Copy the teacher and zero the r columns, so that u(x, t, r) = v_T(x, t) at init."""
        s = cls(**teacher.cfg).to(next(teacher.parameters()).device)
        d = DIM + teacher.emb.dim
        with torch.no_grad():
            s.net[0].weight.zero_()
            s.net[0].weight[:, :d].copy_(teacher.net[0].weight)
            s.net[0].bias.copy_(teacher.net[0].bias)
            for a, b in zip(list(s.net)[1:], list(teacher.net)[1:]):
                if isinstance(a, nn.Linear):
                    a.weight.copy_(b.weight)
                    a.bias.copy_(b.bias)
        return s


NETS = {"VelocityNet": VelocityNet, "FlowMapNet": FlowMapNet}


def save_net(path, net, **extra):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(cls=type(net).__name__, cfg=net.cfg, state=net.state_dict(), **extra), path)


def load_net(path, device):
    d = torch.load(path, map_location=device)
    net = NETS[d["cls"]](**d["cfg"])
    net.load_state_dict(d["state"])
    return net.to(device).eval()


def freeze(net):
    net.eval()
    for p in net.parameters():
        p.requires_grad_(False)
    return net


# ====================================================================================================
# pull / push: the master-formula loss and the positive / negative constructions
#
# Pull/push (contrastive) losses and the positive/negative providers.
#
# Blog form, in x-space: every method moves the student's jump x_hat = X[x_t, u, r] toward
# x_pos and away from x_neg,
#
#     grad_theta = -2 W (d x_hat / d theta)^T (x_pos - x_neg).
#
# Training uses the velocity-space twin `contrastive_loss_v`: with x_pos = X[x_t, u_pos, r]
# and x_neg = X[x_t, u_neg, r] it has the same gradient as the x-space loss with
# W = 1 / (t - r)^2, and it stays finite at r = t.
# ====================================================================================================
def X(x_t, v, t, r):
    """Transport x_t with velocity v from level t to level r."""
    return x_t - (as_col(t, x_t) - as_col(r, x_t)) * v


def renoise(x_r, t, r, eps):
    """Noise a level-r point up to level t >= r on the linear path (the SDE step of the rpd paper):
    a x_r + tau eps with a = (1 - t) / (1 - r) and tau^2 = t^2 - a^2 r^2, which maps p_r to p_t."""
    a = (1 - t) / (1 - r)
    tau = (t * t - (a * r) ** 2).clamp_min(0).sqrt()
    return a * x_r + tau * eps


def level_pred(x_t, v, t, r):
    """E[x_r | x_t] under `renoise`, from the velocity v of the noised path at x_t (Tweedie):
    (x_t + tau^2 s) / a with the score s = -(x_t + (1 - t) v) / t. At r = 0 it is X[x_t, v, 0]."""
    a = (1 - t) / (1 - r)
    tau2 = (t * t - (a * r) ** 2).clamp_min(0)
    return (x_t - tau2 * (x_t + (1 - t) * v) / t) / a


@torch.no_grad()
def drift_field(x, y_pos, y_neg, temps=(0.02, 0.05, 0.2), self_neg=True, kernel="laplace"):
    """Drifting field V of Deng et al. (2026), Alg. 2, with their feature and drift normalization (A.6);
    kernel="gauss" swaps their Laplacian kernel for a Gaussian one at the same bandwidth.

    Per temperature, V_tau = s_pos s_neg (x_pos - x_neg): x_pos and x_neg are kernel-weighted means of the
    positives (data) and negatives (generated samples) around x. Returns the sum over temperatures of the
    normalized drifts, in normalized feature units, and the feature scale S (features are x / S).
    """
    n, d = x.shape
    dist_pos, dist_neg = torch.cdist(x, y_pos), torch.cdist(x, y_neg)
    scale = torch.cat([dist_pos, dist_neg], 1).mean() / d ** 0.5
    if self_neg:
        dist_neg = dist_neg + torch.eye(n, device=x.device) * 1e6
    dist = torch.cat([dist_pos, dist_neg], 1) / scale
    total = torch.zeros_like(x)
    for tau in temps:
        logit = kernel_logit(dist, tau * d ** 0.5, kernel)
        A = (logit.softmax(-1) * logit.softmax(-2)).sqrt()
        A_pos, A_neg = A[:, : len(y_pos)], A[:, len(y_pos):]
        V = (A_neg.sum(1, keepdim=True) * (A_pos @ y_pos) - A_pos.sum(1, keepdim=True) * (A_neg @ y_neg)) / scale
        total += V / (V.pow(2).sum(1).mean() / d).sqrt().clamp_min(1e-12)
    return total, scale


def kernel_logit(dist, h, kernel="laplace"):
    """log k for distances dist at bandwidth h: Laplacian exp(-r / h) (Deng et al.) or Gaussian exp(-r^2 / 2h^2)."""
    return -dist / h if kernel == "laplace" else -dist ** 2 / (2 * h ** 2)


@torch.no_grad()
def kernel_means(x, y_pos, y_neg, tau, scale, kernel="laplace"):
    """Kernel-weighted means of positives and negatives around x (the mean-shift targets of one temperature)."""
    h = scale * tau * x.shape[1] ** 0.5
    w_pos = kernel_logit(torch.cdist(x, y_pos), h, kernel).softmax(-1)
    w_neg = kernel_logit(torch.cdist(x, y_neg), h, kernel).softmax(-1)
    return w_pos @ y_pos, w_neg @ y_neg


def pull_push(y, g, W=None):
    """mean W |y - sg(y + g)|^2: its gradient on y is -2 W g, so descent moves y along g.

    y is whatever the student's parameters produce: a sample x_pred with g = x_pos - x_neg, or a
    velocity u with g = u_pos - u_neg.
    """
    per = ((y - (y + g.detach()).detach()) ** 2).sum(-1, keepdim=True)
    return (per if W is None else W * per).mean()


def contrastive_loss(x_hat, x_pos, x_neg, W):
    """Pull x_hat toward x_pos, push it away from x_neg (x-space, as in the post)."""
    return pull_push(x_hat, x_pos - x_neg, W)


def contrastive_loss_v(u, u_pos, u_neg, w=None):
    """Velocity-space twin: the gradient on u is -2 w (u_pos - u_neg)."""
    return pull_push(u, u_pos - u_neg, w)


# ---------------------------------------------------------------- providers

@torch.no_grad()
def sim_pos_neg(teacher, u, x_t, t, r, n_rk4):
    """Trajectory consistency: positive is the teacher's simulated chord, negative is yourself."""
    return teacher.chord(x_t, t, r, n_rk4), u.detach()


@torch.no_grad()
def mf_J(student, x_t, t, r, tangent, method="jvp", eps_cd=1e-2):
    """J = (t - r) d/dt u(x_t, t, r) along the tangent, detached."""
    gap = t - r
    if method == "jvp":
        _, dudt = torch.func.jvp(
            lambda x, tt, rr: student(x, tt, rr),
            (x_t, t, r),
            (tangent, torch.ones_like(t), torch.zeros_like(r)),
        )
        return gap * dudt
    h_dn = torch.minimum(torch.full_like(t, eps_cd), gap)
    h_up = torch.minimum(torch.full_like(t, eps_cd), 1 - t)
    up = student(x_t + h_up * tangent, t + h_up, r)
    dn = student(x_t - h_dn * tangent, t - h_dn, r)
    J = gap * (up - dn) / (h_up + h_dn).clamp_min(1e-12)
    return torch.where(gap > 0, J, torch.zeros_like(J))


@torch.no_grad()
def mf_pos_neg(student, u, x_t, t, r, v_target, tangent, method="jvp", eps_cd=1e-2):
    """MeanFlow: positive is the target's tangent, negative is the student's self-induced
    tangent u + J (computed from the student itself)."""
    J = mf_J(student, x_t, t, r, tangent, method, eps_cd)
    return v_target, u.detach() + J, J


@torch.no_grad()
def kernel_posterior(x_q, y, t, r, self_mask=True):
    """E[x_r | x_t'] when the level-r law is the empirical distribution of y and x_t' = a x_r + tau eps:
    a Gaussian-kernel mean of y, i.e. the exact denoiser of the samples y. With self_mask, row i excludes
    y_i, the sample x_q[i] was built from."""
    a = (1 - t) / (1 - r)
    var = t ** 2 - a ** 2 * r ** 2
    logit = -((x_q[:, None] - a[:, None] * y[None]) ** 2).sum(-1) / (2 * var)
    if self_mask:
        logit = logit.masked_fill(torch.eye(len(x_q), len(y), dtype=torch.bool, device=x_q.device), -torch.inf)
    return logit.softmax(-1) @ y


@torch.no_grad()
def dmd_pos_neg(v_teacher, fake, x_t, t):
    """DMD: positive is the teacher's tangent, negative is the fake's tangent (learned)."""
    return v_teacher, fake(x_t, t)


# ====================================================================================================
# samplers: few-step chain and renoise samplers of the student
# ====================================================================================================
def time_grid(K):
    return torch.linspace(1.0, 0.0, K + 1).tolist()


@torch.no_grad()
def chain(student, eps, K, return_path=False):
    """Deterministic chord chain on a uniform grid: x_{t'} = x_t - (t - t') u(x_t, t, t')."""
    ts = time_grid(K)
    x, path = eps, [eps]
    for k in range(K):
        x = x - (ts[k] - ts[k + 1]) * student(x, ts[k], ts[k + 1])
        path.append(x)
    return (x, torch.stack(path)) if return_path else x


@torch.no_grad()
def renoise_sampler(student, eps, K, noise_seed=None, return_path=False):
    """Jump to the clean end with u(x_t, t, 0), then renoise to the next level (DMD-style)."""
    ts = time_grid(K)
    x, path = eps, [eps]
    for k in range(K):
        x0 = x - ts[k] * student(x, ts[k], 0.0)
        if k < K - 1:
            z = cpu_randn(x.shape, noise_seed + k).to(x) if noise_seed is not None else torch.randn_like(x)
            x = (1 - ts[k + 1]) * x0 + ts[k + 1] * z
        else:
            x = x0
        path.append(x)
    return (x, torch.stack(path)) if return_path else x


def sample(student, eps, K, sampler, noise_seed=None):
    if sampler == "chain":
        return chain(student, eps, K)
    if sampler == "renoise":
        return renoise_sampler(student, eps, K, noise_seed)
    raise ValueError(sampler)


@torch.no_grad()
def sample_prefix(student, eps, K, sampler):
    """The first K - 1 steps of the K-step sampler: the state x_q at level t_jump = t_{K-1} from which
    the last call x_q - t_jump u(x_q, t_jump, 0) completes a K-step sample (x_q = eps when K = 1)."""
    ts = time_grid(K)
    x = eps
    for k in range(K - 1):
        if sampler == "chain":
            x = x - (ts[k] - ts[k + 1]) * student(x, ts[k], ts[k + 1])
        else:
            x0 = x - ts[k] * student(x, ts[k], 0.0)
            x = (1 - ts[k + 1]) * x0 + ts[k + 1] * torch.randn_like(x)
    return x, ts[K - 1]


def sample_last_grad(student, eps, K, sampler, n_steps=None):
    """The first n_steps (default K) steps of the K-step sampler, ending with the jump from t_{n-1} to 0;
    only that last jump stays in the graph. DMD's backward simulation draws n_steps per update."""
    ts = time_grid(K)
    n = K if n_steps is None else n_steps
    x = eps
    with torch.no_grad():
        for k in range(n - 1):
            if sampler == "chain":
                x = x - (ts[k] - ts[k + 1]) * student(x, ts[k], ts[k + 1])
            else:
                x0 = x - ts[k] * student(x, ts[k], 0.0)
                x = (1 - ts[k + 1]) * x0 + ts[k + 1] * torch.randn_like(x)
    t_last = ts[n - 1]
    return x - t_last * student(x, t_last, 0.0)


def _by_depth(fn, eps, K, n_steps):
    """Apply fn(eps_subset, n) to each group of equal depth n and restore the original order."""
    if n_steps.shape != (len(eps),):
        raise ValueError(f"n_steps must have shape ({len(eps)},), got {tuple(n_steps.shape)}")
    if len(n_steps) == 0 or n_steps.min() < 1 or n_steps.max() > K:
        raise ValueError(f"n_steps values must lie in [1, {K}]")
    order = torch.argsort(n_steps)
    sorted_steps = n_steps[order]
    chunks = [fn(eps[order[sorted_steps == n]], n) for n in range(1, K + 1) if (sorted_steps == n).any()]
    return torch.cat(chunks)[torch.argsort(order)]


@torch.no_grad()
def sample_mixed(student, eps, K, sampler, n_steps):
    """Detached per-sample mixture of n-step samplers, n_steps[i] in {1, ..., K}."""
    return _by_depth(lambda e, n: sample(student, e, n, sampler), eps, K, n_steps)


def sample_mixed_last_grad(student, eps, K, sampler, n_steps):
    """Per-sample mixture of n-step samplers, n_steps[i] in {1, ..., K}, where only each sample's
    final call stays in the graph. Output i follows the same law as sample(student, eps_i, n_steps[i]),
    so it matches a detached rollout of the same mixture."""
    return _by_depth(lambda e, n: sample_last_grad(student, e, n, sampler), eps, K, n_steps)


def sample_full_grad(student, eps, K, sampler):
    """K-step sampler with every student call left in the computation graph."""
    ts = time_grid(K)
    x = eps
    for k in range(K):
        if sampler == "chain":
            x = x - (ts[k] - ts[k + 1]) * student(x, ts[k], ts[k + 1])
        else:
            x0 = x - ts[k] * student(x, ts[k], 0.0)
            if k < K - 1:
                x = (1 - ts[k + 1]) * x0 + ts[k + 1] * torch.randn_like(x)
            else:
                x = x0
    return x


def sample_mixed_full_grad(student, eps, K, sampler, n_steps):
    """Per-sample mixture of n-step samplers, n_steps[i] in {1, ..., K}, with every call differentiated."""
    return _by_depth(lambda e, n: sample_full_grad(student, e, n, sampler), eps, K, n_steps)


# ====================================================================================================
# metrics: SW2, kNN KL, coupling error and others
# ====================================================================================================
CHUNK = 2048


def sqdist(a, b):
    return ((a * a).sum(1)[:, None] + (b * b).sum(1)[None, :] - 2 * a @ b.T).clamp_min(0)


# ---------------------------------------------------------------- distribution level

def sw2(x, y, n_proj=1000, seed=0):
    """Sliced W2: exact 1D W2 per random direction (sorting), averaged in squares."""
    n = min(len(x), len(y))
    x, y = x[:n], y[:n]
    th = cpu_randn((n_proj, x.shape[1]), seed).to(x)
    th = th / th.norm(dim=1, keepdim=True)
    px = (x @ th.T).sort(0).values
    py = (y @ th.T).sort(0).values
    return ((px - py) ** 2).mean().sqrt().item()


def _mean_pair(a, b, fn):
    tot = 0.0
    for i in range(0, len(a), CHUNK):
        tot += fn(sqdist(a[i:i + CHUNK], b)).sum().item()
    return tot / (len(a) * len(b))


def energy(x, y):
    f = lambda d2: d2.sqrt()
    return 2 * _mean_pair(x, y, f) - _mean_pair(x, x, f) - _mean_pair(y, y, f)


def median_dist(x, y, n=1000):
    z = torch.cat([x[:n], y[:n]])
    d = sqdist(z, z).sqrt()
    return d[d > 0].median().item()


def mmd(x, y, factors=(0.25, 0.5, 1.0, 2.0)):
    """Multi-bandwidth RBF MMD^2 (V-statistic), bandwidths relative to the median distance."""
    med = median_dist(x, y)
    bws = [f * med for f in factors]
    k = lambda d2: sum(torch.exp(-d2 / (2 * b * b)) for b in bws)
    return _mean_pair(x, x, k) + _mean_pair(y, y, k) - 2 * _mean_pair(x, y, k)


def nn_dist(a, b):
    ds, idx = [], []
    for i in range(0, len(a), CHUNK):
        m = sqdist(a[i:i + CHUNK], b).min(1)
        ds.append(m.values.sqrt())
        idx.append(m.indices)
    return torch.cat(ds), torch.cat(idx)


def knn_radius(a, k):
    rs = []
    for i in range(0, len(a), CHUNK):
        d2 = sqdist(a[i:i + CHUNK], a)
        rs.append(d2.topk(k + 1, dim=1, largest=False).values[:, -1].sqrt())
    return torch.cat(rs)


def knn_kl(p, q, k=5):
    """kNN estimate of KL(P || Q) in nats from samples p ~ P and q ~ Q (Wang, Kulkarni & Verdu, 2009):
    (d / n) sum_i log(nu_k(i) / rho_k(i)) + log(m / (n - 1)), rho within p, nu from p_i to q."""
    n, m, d = len(p), len(q), p.shape[1]
    rho = knn_radius(p, k)
    nu = torch.cat([sqdist(p[i:i + CHUNK], q).topk(k, dim=1, largest=False).values[:, -1].sqrt()
                    for i in range(0, n, CHUNK)])
    return (d * torch.log(nu.clamp_min(1e-12) / rho.clamp_min(1e-12)).mean() + np.log(m / (n - 1))).item()


def _coverage(p, ref, radius):
    r2 = (radius ** 2)[None, :]
    inside = []
    for i in range(0, len(p), CHUNK):
        inside.append((sqdist(p[i:i + CHUNK], ref) <= r2).any(1))
    return torch.cat(inside).float().mean().item()


def precision_recall(real, fake, k=5):
    """k-NN precision/recall (Kynkaanniemi et al., 2019)."""
    return _coverage(fake, real, knn_radius(real, k)), _coverage(real, fake, knn_radius(fake, k))


def mode_balance(x, ref, ref_lab):
    _, idx = nn_dist(x, ref)
    p = ref_lab[idx].float().mean().item()
    return abs(p - 0.5), p


def off_manifold(x, ref, tau):
    d, _ = nn_dist(x, ref)
    return (d > tau).float().mean().item()


def tau_from_refs(A, B, q=0.99):
    d, _ = nn_dist(B, A)
    return torch.quantile(d.float().cpu(), q).item()


def dist_metrics(x, ref, ref_lab, tau, ecfg, seed=0):
    ne, nk = min(ecfg.energy_n, len(x)), min(ecfg.knn_n, len(x))
    prec, rec = precision_recall(ref[:nk], x[:nk], ecfg.knn_k)
    imb, p_up = mode_balance(x, ref, ref_lab)
    return dict(
        kl_rev=knn_kl(x[:nk], ref[:nk]),
        kl_fwd=knn_kl(ref[:nk], x[:nk]),
        sw2=sw2(x, ref, ecfg.n_proj, seed),
        energy=energy(x[:ne], ref[:ne]),
        mmd=mmd(x[:ne], ref[:ne], ecfg.mmd_factors),
        precision=prec,
        recall=rec,
        mode_imbalance=imb,
        p_upper=p_up,
        off_manifold=off_manifold(x, ref, tau),
    )


# ---------------------------------------------------------------- trajectory level

def coupling_error(x, end):
    return ((x - end) ** 2).sum(-1).mean().item()


@torch.no_grad()
def chord_triangle(u_fn, S, grid_t, v_fn=None):
    """Relative chord error on the (t, r) triangle.

    S[i] are states on teacher trajectories at grid_t[i] (ascending), so the exact chord
    between grid times is (S[i] - S[j]) / (t_i - t_j). err[i, j] for j < i; err[i, i] is the
    diagonal (instantaneous) error against v_fn when given.
    """
    gt = [float(v) for v in grid_t]
    n = len(gt) - 1
    err = np.full((n + 1, n + 1), np.nan)
    for i in range(1, n + 1):
        x, t = S[i], gt[i]
        rs = torch.tensor(gt[:i], device=x.device, dtype=x.dtype)
        xx = x.repeat(i, 1)
        rr = rs.repeat_interleave(len(x))[:, None]
        u = u_fn(xx, t, rr).reshape(i, len(x), x.shape[1])
        ustar = (x[None] - S[:i]) / (t - rs)[:, None, None]
        e = ((u - ustar) ** 2).sum(-1).mean(1) / (ustar ** 2).sum(-1).mean(1)
        err[i, :i] = e.cpu().numpy()
        if v_fn is not None:
            ud, v = u_fn(x, t, t), v_fn(x, t)
            err[i, i] = (((ud - v) ** 2).sum(-1).mean() / (v ** 2).sum(-1).mean()).item()
    return err


def triangle_summary(err):
    n = err.shape[0]
    cells = np.array([err[i, j] for i in range(1, n) for j in range(i)])
    rpos = np.array([err[i, j] for i in range(2, n) for j in range(1, i)])
    diag = np.array([err[i, i] for i in range(1, n)])
    return dict(
        tri_mean=float(np.nanmean(cells)),
        tri_r0=float(np.nanmean(err[1:, 0])),
        tri_rpos=float(np.nanmean(rpos)) if len(rpos) else float("nan"),
        tri_diag=float(np.nanmean(diag)) if np.isfinite(diag).any() else float("nan"),
    )


@torch.no_grad()
def composition(u_fn, S, grid_t, n_triples=60, seed=0):
    """Semigroup consistency: jump t->s->r versus t->r directly, on grid states."""
    gt = [float(v) for v in grid_t]
    n = len(gt) - 1
    triples = [c for c in itertools.combinations(range(n + 1), 3)]
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(triples), size=min(n_triples, len(triples)), replace=False)
    errs = []
    for p in pick:
        j, k, i = triples[p]
        x, t, s, r = S[i], gt[i], gt[k], gt[j]
        xs = x - (t - s) * u_fn(x, t, s)
        two = xs - (s - r) * u_fn(xs, s, r)
        one = x - (t - r) * u_fn(x, t, r)
        errs.append(((two - one) ** 2).sum(-1).mean().item())
    return float(np.mean(errs))


# ====================================================================================================
# teacher: flow-matching teacher, its caches and evaluation
# ====================================================================================================
TDIR = ART / "teacher"


class Teacher:
    """Frozen flow-matching teacher with ODE utilities."""

    def __init__(self, net):
        self.net = freeze(net)

    @torch.no_grad()
    def v(self, x, t):
        return self.net(x, t)

    __call__ = v

    @torch.no_grad()
    def flow(self, x, t, r, n=32, return_path=False):
        """RK4 from level t down to level r with n steps (per-sample t and r allowed)."""
        t, r = as_col(t, x), as_col(r, x)
        h = (t - r) / n
        tau, path = t.clone(), [x]
        for _ in range(n):
            k1 = self.net(x, tau)
            k2 = self.net(x - 0.5 * h * k1, tau - 0.5 * h)
            k3 = self.net(x - 0.5 * h * k2, tau - 0.5 * h)
            k4 = self.net(x - h * k3, tau - h)
            x = x - h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
            tau = tau - h
            if return_path:
                path.append(x)
        return (x, torch.stack(path)) if return_path else x

    @torch.no_grad()
    def chord(self, x, t, r, n=32):
        """Exact (simulated) average velocity over [r, t]; the tangent when t - r is tiny."""
        t, r = as_col(t, x), as_col(r, x)
        gap = t - r
        u = (x - self.flow(x, t, r, n)) / gap.clamp_min(1e-12)
        return torch.where(gap < 1e-3, self.net(x, t), u)

    @torch.no_grad()
    def euler(self, eps, K, return_path=False):
        ts = torch.linspace(1.0, 0.0, K + 1).tolist()
        x, path = eps, [eps]
        for k in range(K):
            x = x - (ts[k] - ts[k + 1]) * self.net(x, ts[k])
            path.append(x)
        return (x, torch.stack(path)) if return_path else x

    @torch.no_grad()
    def rk4(self, eps, n=100, return_path=False):
        return self.flow(eps, 1.0, 0.0, n, return_path)


def load_teacher(device):
    return Teacher(load_net(TDIR / "teacher.pt", device))


def teacher_sha():
    """Identifies the teacher a run was distilled from, so stale runs can be told apart."""
    return hashlib.sha1((TDIR / "teacher.pt").read_bytes()).hexdigest()[:12]


def train_teacher(preset, device):
    cfg = preset.teacher
    moons = make_data(preset.data.noise, device)
    net = VelocityNet(cfg.width, cfg.depth, cfg.n_freq).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=cfg.lr, weight_decay=0.0)
    ema = EMA(net, cfg.ema)
    log = CSVLog(TDIR / "train_log.csv")
    t0 = time.time()
    for step in range(cfg.steps):
        lr = cosine_lr(step, cfg.steps, cfg.lr, cfg.lr_min, cfg.warmup)
        for g in opt.param_groups:
            g["lr"] = lr
        x0 = moons.sample(cfg.batch)
        eps = torch.randn_like(x0)
        t = torch.rand(cfg.batch, 1, device=device)
        xt = (1 - t) * x0 + t * eps
        loss = ((net(xt, t) - (eps - x0)) ** 2).sum(-1).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), cfg.clip)
        opt.step()
        ema.update(net)
        if step % cfg.log_every == 0 or step == cfg.steps - 1:
            log.add(step=step, loss=loss.item(), lr=lr, elapsed=time.time() - t0)
            print(f"[teacher] step {step:6d} loss {loss.item():.4f} ({time.time() - t0:.0f}s)", flush=True)
    log.save()
    save_net(TDIR / "teacher.pt", ema.shadow, preset=preset.name)
    return Teacher(ema.shadow)


def build_caches(teacher, preset, device):
    """Ground truth reused by every student: reference samples, ODE endpoints, and states on
    teacher trajectories at the triangle grid times."""
    e = preset.eval
    eps = cpu_randn((e.n, DIM), 100).to(device)
    torch.save(dict(eps=eps.cpu(), x=teacher.rk4(eps, preset.teacher.ref_rk4).cpu()),
               TDIR / "teacher_ref.pt")
    eps_c = cpu_randn((e.coupling_n, DIM), 101).to(device)
    torch.save(dict(eps=eps_c.cpu(), end=teacher.rk4(eps_c, e.coupling_rk4).cpu()),
               TDIR / "gt_coupling.pt")
    eps_t = cpu_randn((e.tri_states, DIM), 102).to(device)
    _, path = teacher.flow(eps_t, 1.0, 0.0, e.tri_rk4, return_path=True)
    stride = e.tri_rk4 // e.tri_nt
    S = path[::stride].flip(0)
    torch.save(dict(S=S.cpu(), grid_t=torch.linspace(0, 1, e.tri_nt + 1), eps=eps_t.cpu()),
               TDIR / "gt_triangle.pt")


def evaluate_teacher(teacher, preset, device):
    e = preset.eval
    moons = make_data(preset.data.noise, device)
    refs = load_refs(moons, preset.data.n_ref)
    A, A_lab, B = refs["A"][:e.n], refs["A_lab"][:e.n], refs["B"][:e.n]
    tau = tau_from_refs(A, B)
    ref = torch.load(TDIR / "teacher_ref.pt")
    x_ref, eps = ref["x"].to(device), ref["eps"].to(device)
    out = dict(tau=tau, floor=dist_metrics(B, A, A_lab, tau, e),
               rk4=dist_metrics(x_ref, A, A_lab, tau, e))
    out["euler"] = {}
    n_show = min(2000, e.n)
    show = dict(data=A[:n_show].cpu().numpy(), rk4=x_ref[:n_show].cpu().numpy())
    for K in e.teacher_ks:
        x = teacher.euler(eps, K)
        d = dist_metrics(x, A, A_lab, tau, e)
        d["sw2_teacher"] = sw2(x, x_ref, e.n_proj)
        out["euler"][str(K)] = d
        show[f"euler{K}"] = x[:n_show].cpu().numpy()
    out["accept"] = bool(out["rk4"]["sw2"] <= 2 * out["floor"]["sw2"])
    out["preset"], out["teacher_sha"] = preset.name, teacher_sha()
    save_json(TDIR / "eval.json", out)
    np.savez_compressed(TDIR / "samples.npz", **show)
    print(f"[teacher] SW2 rk4 {out['rk4']['sw2']:.4f}  floor {out['floor']['sw2']:.4f}  "
          f"euler1 {out['euler']['1']['sw2']:.4f}  accept={out['accept']}", flush=True)
    return out


# ====================================================================================================
# construct: x_pred, x_pos, x_neg on fixed anchor points, for the figures and animations
# ====================================================================================================
N_ANCHOR = 8
S_OFF, T_MF, R_ALL, U_QUERY = 0.6, 0.6, 0.3, 0.5   # levels; DMD queries at t' = r + (1 - r) U_QUERY
FIELDS = ("start", "x_pred", "query", "x_pos", "x_neg")


def anchors(moons, device):
    """Fixed data points and the two noises of the construction."""
    data, _ = moons.fixed(N_ANCHOR, seed=951)
    return data.to(device), cpu_randn((N_ANCHOR, DIM), 952).to(device), cpu_randn((N_ANCHOR, DIM), 953).to(device)


def supported(variant):
    return parse_variant(variant)[0] in ("dmd", "mf", "drift")


def construction(variant, student, fake, teacher, data, eps1, eps2, drift=None):
    """Start state, moved point x_pred, query state, x_pos and x_neg on the anchors (all detached).

    drift: dict(pos=fixed data positives, n=number of generated negatives, tau=kernel temperature); the
    drift points are the kernel-weighted means of positives and negatives around the one-step sample.
    """
    est, policy, level = parse_variant(variant)
    n, dev = len(eps1), eps1.device
    col = lambda v: torch.full((n, 1), v, device=dev)
    r = col(0.0 if level == "r0" else R_ALL)
    with torch.random.fork_rng(devices=[] if dev.type == "cpu" else [dev]):
        torch.manual_seed(7)
        if est == "drift":
            with torch.no_grad():
                x_pred = sample(student, eps1, 1, "chain")
                y_neg = sample(student, cpu_randn((drift["n"], DIM), 955).to(dev), 1, "chain")
                y_pos = drift["pos"]
                d = x_pred.shape[1]
                scale = torch.cat([torch.cdist(y_neg, y_pos), torch.cdist(y_neg, y_neg)], 1).mean() / d ** 0.5
                x_pos, x_neg = kernel_means(x_pred, y_pos, y_neg, drift["tau"], scale, drift.get("kernel", "laplace"))
            return dict(start=eps1, x_pred=x_pred, query=x_pred, x_pos=x_pos, x_neg=x_neg, y_neg=y_neg)
        if est == "dmd":
            if policy == "on":
                x_s, s0 = sample_prefix(student, eps1, 2, native_sampler(level))
                s = col(s0)
            else:
                s = col(S_OFF)
                x_s = (1 - s) * data + s * eps1
            with torch.no_grad():
                x_pred = x_s - (s - r) * student(x_s, s, r)
                t = r + (1 - r) * U_QUERY
                x_q = renoise(x_pred, t, r, eps2)
                x_pos = level_pred(x_q, teacher.v(x_q, t), t, r)
                x_neg = level_pred(x_q, fake(x_q, t, r), t, r)
            return dict(start=x_s, x_pred=x_pred, query=x_q, x_pos=x_pos, x_neg=x_neg)
        x0 = sample(student, eps2, 2, native_sampler(level)) if policy == "on" else data
    t = col(T_MF)
    x_t = (1 - t) * x0 + t * eps1
    with torch.no_grad():
        u, v_T = student(x_t, t, r), teacher.v(x_t, t)
    J = mf_J(student, x_t, t, r, v_T)
    return dict(start=x_t, x_pred=X(x_t, u, t, r), query=x_t, x_pos=X(x_t, v_T, t, r), x_neg=X(x_t, u + J, t, r))


# ====================================================================================================
# train_student: distil the teacher into a flow-map student with one pull / push loss
#
# Distil the teacher into a flow-map student with one pull/push loss.
#
#     python assess.py train --variant mf-on-all --seed 0 --preset smoke
#
# Variant = {est}-{policy}-{level}:
#   est    sim (simulated chord) | mf (self-induced negative) | dmd (auxiliary negative) | hybrid
#   policy off (x_0 from data) | on (x_0 from the student's own few-step samples)
#   level  r0 (r = 0 only) | all (r anywhere in [0, t])
#
# Every training step draws one rollout depth K ~ Uniform{1, ..., k_roll} and runs five stages:
#   1. rollout for the auxiliary network: n_fm fresh K-step rollouts, one FM step on v_phi each (dmd, hybrid)
#   2. rollout for x_pred: the quantity y the student's parameters move
#   3. g: the pull/push displacement x_pos - x_neg (or its velocity twin) and its weight W
#   4. loss for u_theta: pull_push(y, g, W) = W |y - sg(y + g)|^2
#   5. backward
#
# DMD (dmd-on-r0) builds x_pred as the first K - 1 steps of the K-step sampler (detached) plus its last jump
# at the grid level t_jump, so x_pred has the law of the K-step samples v_phi is fitted to. The DMD
# estimators differ only in where g = t (v_phi - v_T) is queried:
#   native    at noise(sg(x_pred), t), t ~ U(tmin, tmax)
#   grid      at the sampler state x_q itself, t = t_jump
#   surrogate the paper's x-space form: renoise a K-step sample to a random t and move one jump from there
# ====================================================================================================
def get_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--variant", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--preset", default="full")
    p.add_argument("--device", default="auto")
    p.add_argument("--tag", default="")
    p.add_argument("--target", default="teacher", choices=["teacher", "vcond"],
                   help="positive velocity for mf: teacher tangent or conditional velocity")
    p.add_argument("--tangent", default="teacher", choices=["teacher", "vcond"],
                   help="direction of d/dt in J")
    p.add_argument("--j-method", default="jvp", choices=["jvp", "fd"])
    p.add_argument("--dmd-jac", default="native", choices=["native", "grid", "teacher-ode", "surrogate"],
                   help="native: x_pred = x_r is the student's jump to level r and g is queried at "
                        "renoise(sg(x_r), t' > r), every dmd variant; grid / teacher-ode (dmd-on-r0 only): g at "
                        "the jump's input state; surrogate: the paper's renoised K-step sample")
    p.add_argument("--dmd-sampler", default=None, choices=["renoise", "chain"],
                   help="sampler of the on-policy dmd prefix (default: the level's own sampler)")
    p.add_argument("--snap-count", type=int, default=None, help="sample-cloud snapshots over training")
    p.add_argument("--snap-n", type=int, default=None, help="samples per snapshot cloud")
    p.add_argument("--weight", default="unit", choices=["unit", "native", "dmd2norm"])
    p.add_argument("--init", default="teacher", choices=["teacher", "random"])
    p.add_argument("--k-roll", type=int, default=None)
    p.add_argument("--n-fm", type=int, default=None)
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--batch", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--fake-lr", type=float, default=None)
    p.add_argument("--diag-fake", action="store_true",
                   help="train a fake alongside an mf run only to log |v_ind - v_phi|^2")
    p.add_argument("--milestones", action="store_true",
                   help="save raw checkpoints at early steps (V4/V5 exports)")
    p.add_argument("--root", default=None, help="directory for the run (default artifacts/runs)")
    p.add_argument("--mf-neg", default="self", choices=["self", "frozen", "noJ"],
                   help="ablation of the mf negative: sg(u) + J, sg(u_init) + J, or sg(u)")
    p.add_argument("--dmd-neg", default="fake", choices=["fake", "kernel"],
                   help="renoise DMD negative: the fake network's E[x_r | x_t'], or the exact one under the "
                        "empirical distribution of the batch (a Gaussian-kernel mean, no fake)")
    p.add_argument("--drift-n", type=int, default=None,
                   help="drift and kernel-negative DMD: generated samples per step, which are also the "
                        "negatives (default: batch)")
    p.add_argument("--drift-npos", type=int, default=None, help="drift: data samples (positives) per step")
    p.add_argument("--drift-temps", type=float, nargs="+", default=[0.02, 0.05, 0.2])
    p.add_argument("--drift-kernel", default="laplace", choices=["laplace", "gauss"])
    return p.parse_args(argv)


@dataclass
class Pred:
    """Output of stage 2.

    y is what the student's parameters produce and the loss moves: the sample x_pred itself (native and
    grid dmd), or the velocity u = u_theta(x_q, t, r) whose jump X[x_q, u, r] is x_pred. The velocity form
    has the same gradient as x-space with W = 1 / (t - r)^2 and stays finite at r = t.
    x_g is where g is queried, at level t.
    """
    y: torch.Tensor
    t: torch.Tensor
    r: torch.Tensor
    x_g: torch.Tensor
    v_cond: torch.Tensor = None
    x_roll: torch.Tensor = None


def dmd_weight(kind, t, r, v_T, v_cond):
    """Velocity-space weight w = W (t - r)^2 of the surrogate DMD loss.

    native: W = w_t / (2t) with w_t = (1 - t) / t, the paper's DMD weight.
    dmd2norm: W = 1 / (2c), c = mean |x_r - x_pos|, where x_r = X[x_t, v_cond, r] is the clean sample's
    own point at level r; at r = 0 this is DMD2's |latents - pred_real_image| (sd_guidance.py).
    """
    if kind == "unit":
        return None
    if kind == "native":
        return 0.5 * (1 - t) * ((t - r) / t) ** 2
    c = ((t - r) * (v_T - v_cond)).abs().mean(-1, keepdim=True).clamp_min(1e-3)
    return (t - r) ** 2 / (2 * c)


def native_jac_terms(kind, t, x_hat, x_t, v_T, v_phi):
    """Displacement g and weight W of the DMD loss W |x_hat - sg(x_hat + g)|^2 on a sample x_hat,
    where x_pos - x_neg = X[x_t, v_T, 0] - X[x_t, v_phi, 0].

    unit: the gradient on x_hat is 2 (v_T - v_phi), the velocity difference itself.
    native: the gradient on x_hat is w_t (v_T - v_phi), the paper's DMD gradient.
    dmd2norm: DMD2's normalised displacement.
    """
    g_x = t * (v_phi - v_T)
    if kind == "unit":
        return g_x / t, 1.0
    if kind == "native":
        return g_x, (1 - t) / (2 * t * t)
    c = (x_hat.detach() - (x_t - t * v_T)).abs().mean(-1, keepdim=True).clamp_min(1e-3)
    return g_x / c, 0.5


def cost_per_sample(est, policy, args, cfg):
    """Expected network calls per training sample; the diagnostic fake of --diag-fake is not counted."""
    kernel_neg = est == "dmd" and args.dmd_neg == "kernel"
    use_fake = est in ("dmd", "hybrid") and not kernel_neg
    uses_teacher_mf = args.target == "teacher" or args.tangent == "teacher"
    teacher_nfe = {"sim": 4 * cfg.sim_rk4, "mf": int(uses_teacher_mf), "dmd": 1, "hybrid": 1, "drift": 0}[est]
    mf_like = est in ("mf", "hybrid")
    roll = (cfg.k_roll + 1) / 2
    fake_roll = cfg.n_fm * roll if use_fake else 0
    if est == "drift" or kernel_neg:
        student_fwd = roll if policy == "on" else 1
    elif est == "dmd" and args.dmd_jac == "native":
        jump = roll if policy == "on" else 1
        student_fwd = cfg.n_fm * jump + jump
    elif est == "dmd" and args.dmd_jac == "grid":
        student_fwd = fake_roll + roll
    else:
        student_fwd = fake_roll + 1 + (roll if policy == "on" else 0)
        if mf_like and args.j_method == "fd":
            student_fwd += 2
    return dict(
        teacher_nfe=teacher_nfe,
        student_fwd=student_fwd,
        student_jvp=int(mf_like and args.j_method == "jvp"),
        fake_fwd=int(use_fake),
        fake_train_steps=cfg.n_fm if use_fake else 0,
    )


def train_student(args):
    preset = get_preset(args.preset)
    cfg = preset.student
    cfg = replace(cfg, **{k: v for k, v in dict(steps=args.steps, batch=args.batch, k_roll=args.k_roll,
                                                 n_fm=args.n_fm, lr=args.lr,
                                                 fake_lr=args.fake_lr, snap_count=args.snap_count,
                                                 snap_n=args.snap_n).items()
                          if v is not None})
    device = pick_device(args.device)
    seed_all(args.seed)
    est, policy, level = parse_variant(args.variant)
    renoise_dmd = est == "dmd" and args.dmd_jac == "native"
    sample_dmd = est == "dmd" and args.dmd_jac in ("grid", "teacher-ode")
    if sample_dmd:
        assert policy == "on" and level == "r0", f"{args.dmd_jac} DMD is defined for dmd-on-r0"
    if renoise_dmd:
        assert args.weight == "unit", "renoise DMD implements the unit weight"
    if est == "drift":
        assert policy == "on" and level == "r0", "drift moves the student's own clean samples: use drift-on-r0"
    kernel_neg = est == "dmd" and args.dmd_neg == "kernel"
    if kernel_neg:
        assert renoise_dmd and level == "r0", "the kernel negative needs one level for the whole batch: dmd-*-r0"
    drift_n = args.drift_n or cfg.batch
    drift_npos = args.drift_npos or drift_n
    name = run_name(args.variant, args.tag)
    out = (Path(args.root) if args.root else ART / "runs") / name / f"seed{args.seed}"
    out.mkdir(parents=True, exist_ok=True)
    for stale in ("config.json", "eval.json"):
        (out / stale).unlink(missing_ok=True)
    shutil.rmtree(out / "milestones", ignore_errors=True)

    teacher = load_teacher(device)
    moons = make_data(preset.data.noise, device)
    if args.init == "teacher":
        student = FlowMapNet.from_teacher(teacher.net)
    else:
        student = FlowMapNet(**teacher.net.cfg).to(device)
    student.train()
    aux_fake = est in ("dmd", "hybrid") and not kernel_neg
    diag = args.diag_fake and not aux_fake
    fake = None
    if aux_fake or diag:
        # The renoise-DMD fake estimates E[x_r | x_t'] for the student's level-r marginal, so like the
        # student it is conditioned on r; at init it equals the teacher, so x_neg = x_pos.
        fake = (FlowMapNet.from_teacher(teacher.net) if renoise_dmd else copy.deepcopy(teacher.net)).train()
        for p in fake.parameters():
            p.requires_grad_(True)
        opt_f = torch.optim.Adam(fake.parameters(), lr=cfg.fake_lr)
    opt = torch.optim.Adam(student.parameters(), lr=cfg.lr)
    ema = EMA(student, cfg.ema)
    sampler = native_sampler(level)
    roll_sampler = args.dmd_sampler or sampler
    student0 = copy.deepcopy(student).eval() if args.mf_neg == "frozen" else None

    ref = torch.load(TDIR / "teacher_ref.pt")
    x_tref = ref["x"][:cfg.quick_n].to(device)
    gtc = torch.load(TDIR / "gt_coupling.pt")
    eps_c, end_c = gtc["eps"][:cfg.quick_n].to(device), gtc["end"][:cfg.quick_n].to(device)
    eps_snap = cpu_randn((cfg.snap_n, DIM), 200).to(device)
    # The anchor construction needs a fake that is conditioned on r, i.e. renoise DMD, or none (mf, drift).
    demo = supported(args.variant) and (est in ("mf", "drift") or (renoise_dmd and not kernel_neg))
    anchor_set = anchors(moons, device) if demo else None
    drift_vis = (dict(pos=moons.fixed(drift_npos, seed=954)[0], n=drift_n, tau=max(args.drift_temps),
                      kernel=args.drift_kernel)
                 if est == "drift" else None)
    snap_steps = sorted(set(np.linspace(0, cfg.steps, cfg.snap_count).astype(int).tolist()))
    want_ms = args.milestones and args.seed == 0 and name in MILESTONE_RUNS
    milestones = sorted({m for m in cfg.milestones if m <= cfg.steps} | {cfg.steps}) if want_ms else []

    def draw_t(B):
        if est == "dmd":
            return cfg.dmd_tmin + (cfg.dmd_tmax - cfg.dmd_tmin) * torch.rand(B, 1, device=device)
        return torch.rand(B, 1, device=device)

    def draw_r(t):
        if level == "r0":
            return torch.zeros_like(t)
        c = torch.rand_like(t)
        r = torch.rand_like(t) * t
        r = torch.where(c < cfg.frac_diag + cfg.frac_zero, torch.zeros_like(t), r)
        return torch.where(c < cfg.frac_diag, t, r)

    @torch.no_grad()
    def rollout(B, K):
        """Detached K-step student samples."""
        return sample(student, torch.randn(B, DIM, device=device), K, roll_sampler)

    def fm_step(xs):
        tf = torch.rand(len(xs), 1, device=device)
        ef = torch.randn_like(xs)
        lf = ((fake((1 - tf) * xs + tf * ef, tf) - (ef - xs)) ** 2).sum(-1).mean()
        opt_f.zero_grad(set_to_none=True)
        lf.backward()
        torch.nn.utils.clip_grad_norm_(fake.parameters(), cfg.clip)
        opt_f.step()

    def dmd_jump(B, K):
        """x_r: the student's jump to level r from a state x_s, which is its own K-step sampler state
        (on-policy) or noised data (off-policy). r = 0 at level r0; otherwise r is 0 or U(0, s)."""
        if policy == "on":
            x_s, s0 = sample_prefix(student, torch.randn(B, DIM, device=device), K, roll_sampler)
            s = torch.full((B, 1), s0, device=device)
        else:
            s = draw_t(B)
            x_s = (1 - s) * moons.sample(B) + s * torch.randn(B, DIM, device=device)
        r = torch.zeros_like(s)
        if level == "all":
            # r stays below 0.95: as r -> 1 the renoise gain a = (1 - t') / (1 - r) underflows in float32.
            r = torch.where(torch.rand_like(s) < 1 / 3, r, (torch.rand_like(s) * s).clamp(max=0.95))
        return x_s - (s - r) * student(x_s, s, r), r

    def renoise_query(x_r, r):
        """Detached query state x_t' = renoise(x_r) at t' ~ r + (1 - r) U(tmin, tmax)."""
        t = r + (1 - r) * (cfg.dmd_tmin + (cfg.dmd_tmax - cfg.dmd_tmin) * torch.rand_like(r))
        return renoise(x_r.detach(), t, r, torch.randn_like(x_r)), t

    def fake_step(x_r, r):
        """Denoising score matching for the student's level-r marginal: fake's E[x_r | x_t'] -> x_r."""
        x_q, t = renoise_query(x_r, r)
        lf = ((level_pred(x_q, fake(x_q, t, r), t, r) - x_r) ** 2).sum(-1).mean()
        opt_f.zero_grad(set_to_none=True)
        lf.backward()
        torch.nn.utils.clip_grad_norm_(fake.parameters(), cfg.clip)
        opt_f.step()

    def untimed(fn, *a):
        """Run work that is not part of the method (the diagnostic fake) outside the step timer."""
        nonlocal aside
        sync(device)
        a0 = time.time()
        res = fn(*a)
        sync(device)
        aside += time.time() - a0
        return res

    # ------------------------------------------------------------ stage 2
    def predict(B, K):
        if est == "drift":
            # A fresh K-step sample whose last step carries the gradient; drift builds its points around it.
            x_q, t_jump = sample_prefix(student, torch.randn(drift_n, DIM, device=device), K, roll_sampler)
            tj = torch.full((drift_n, 1), t_jump, device=device)
            x_pred = x_q - tj * student(x_q, tj, torch.zeros_like(tj))
            return Pred(y=x_pred, t=torch.ones_like(tj), r=torch.zeros_like(tj), x_g=x_pred)
        if renoise_dmd:
            x_r, r = dmd_jump(B, K)
            x_q, t = renoise_query(x_r, r)
            return Pred(y=x_r, t=t, r=r, x_g=x_q)
        if est == "dmd" and args.dmd_jac == "teacher-ode":
            t = draw_t(B)
            x_q = teacher.flow(torch.randn(B, DIM, device=device), 1.0, t, cfg.sim_rk4)
            x_pred = x_q - t * student(x_q, t, torch.zeros_like(t))
            return Pred(y=x_pred, t=t, r=torch.zeros_like(t), x_g=x_q)
        if sample_dmd:
            # x_pred = last jump of a K-step sample: same law as the K-step rollouts v_phi was fitted to.
            x_q, t_jump = sample_prefix(student, torch.randn(B, DIM, device=device), K, roll_sampler)
            tj = torch.full((B, 1), t_jump, device=device)
            x_pred = x_q - tj * student(x_q, tj, torch.zeros_like(tj))
            if args.dmd_jac == "native":
                t = draw_t(B)
                x_g = (1 - t) * x_pred.detach() + t * torch.randn_like(x_pred)
            else:
                t, x_g = tj, x_q
            return Pred(y=x_pred, t=t, r=torch.zeros_like(t), x_g=x_g)
        # Velocity family: a state from K-step student samples (on) or data (off), then one jump.
        x_roll = rollout(B, K) if policy == "on" else None
        x0 = x_roll if policy == "on" else moons.sample(B)
        eps = torch.randn_like(x0)
        t = draw_t(B)
        r = draw_r(t)
        x_q = (1 - t) * x0 + t * eps
        return Pred(y=student(x_q, t, r), t=t, r=r, x_g=x_q, v_cond=eps - x0, x_roll=x_roll)

    # ------------------------------------------------------------ stage 3
    def compute_g(P, step):
        """Displacement g and weight W on P.y, plus the squared velocity gap for logging."""
        t, r, x_g, info = P.t, P.r, P.x_g, {}
        if est == "drift":
            # Positives: fresh data; negatives: this batch of generated samples (each excluding itself).
            # The loss |x/S - sg(x/S + V)|^2 of the paper is pull_push(x, S V, 1/S^2).
            x = P.y.detach()
            V, scale = drift_field(x, moons.sample(drift_npos), x, args.drift_temps, kernel=args.drift_kernel)
            return V * scale, 1.0 / scale ** 2, (V ** 2).sum(-1), info
        if renoise_dmd:
            # pos / neg: the teacher's and the fake's E[x_r | x_t'], both built at the renoised x_t'.
            with torch.no_grad():
                x_pos = level_pred(x_g, teacher.v(x_g, t), t, r)
                if kernel_neg:
                    # The batch's own samples, each query excluding the sample it was noised from.
                    x_neg = kernel_posterior(x_g, P.y.detach(), t, r)
                else:
                    x_neg = level_pred(x_g, fake(x_g, t, r), t, r)
            g = (x_pos - x_neg) / t
            return g, None, (g ** 2).sum(-1), info
        if sample_dmd:
            with torch.no_grad():
                v_T, v_phi = teacher.v(x_g, t), fake(x_g, t)
            g, W = native_jac_terms(args.weight, t, P.y, x_g, v_T, v_phi)
            return g, W, ((v_T - v_phi) ** 2).sum(-1), info
        u = P.y
        if est == "sim":
            u_pos, u_neg = sim_pos_neg(teacher, u, x_g, t, r, cfg.sim_rk4)
            return u_pos - u_neg, None, ((u_pos - u_neg) ** 2).sum(-1), info
        if est == "dmd":
            v_T = teacher.v(x_g, t)
            u_pos, u_neg = dmd_pos_neg(v_T, fake, x_g, t)
            return u_pos - u_neg, dmd_weight(args.weight, t, r, v_T, P.v_cond), ((u_pos - u_neg) ** 2).sum(-1), info
        # mf and hybrid: the negative is the student's self-induced tangent sg(u) + J
        need_vt = args.target == "teacher" or args.tangent == "teacher" or est == "hybrid"
        v_T = teacher.v(x_g, t) if need_vt else None
        v_target = v_T if args.target == "teacher" else P.v_cond
        tangent = v_T if args.tangent == "teacher" else P.v_cond
        u_pos, u_neg, J = mf_pos_neg(student, u, x_g, t, r, v_target, tangent, args.j_method, cfg.eps_cd)
        if args.mf_neg == "noJ":
            u_neg = u.detach()
        elif args.mf_neg == "frozen":
            with torch.no_grad():
                u_neg = student0(x_g, t, r) + J
        if args.diag_fake and step % cfg.log_every == 0:
            with torch.no_grad():
                info["v_ind_vs_v_phi"] = ((u_neg - fake(x_g, t)) ** 2).sum(-1).mean().item()
        g = u_pos - u_neg
        if est == "hybrid":
            # Gradients add, so lambda * (dmd term) folds into one displacement with unit weight.
            inside = ((t > cfg.dmd_tmin) & (t < cfg.dmd_tmax)).float()
            up2, un2 = dmd_pos_neg(v_T, fake, x_g, t)
            w2 = dmd_weight(args.weight, t, r, v_T, P.v_cond)
            w2 = inside if w2 is None else w2 * inside
            g = g + cfg.hybrid_lambda * w2 * (up2 - un2)
        return g, None, ((u_pos - u_neg) ** 2).sum(-1), info

    @torch.no_grad()
    def quick_eval(step):
        net = ema.shadow
        x1 = sample(net, eps_c, 1, "chain")
        x4 = sample(net, eps_c, 4, sampler, noise_seed=300)
        return dict(step=step, sw2_k1=sw2(x1, x_tref, cfg.quick_proj),
                    sw2_k4=sw2(x4, x_tref, cfg.quick_proj),
                    coupling_k1=coupling_error(x1, end_c))

    log, qlog = CSVLog(out / "train_log.csv"), CSVLog(out / "quick_log.csv")
    snaps1, snaps2, snaps4, snap_at = [], [], [], []
    cons = {k: [] for k in FIELDS + (("y_neg",) if est == "drift" else ())}
    ms_hist = []
    sync(device)
    t_start = t_mark = time.time()
    n_mark, aside, diverged_at = 0, 0.0, None
    for step in range(cfg.steps + 1):
        if step in snap_steps or step % cfg.quick_every == 0 or step == cfg.steps or step in milestones:
            sync(device)
            a0 = time.time()
            if step in snap_steps:
                with torch.no_grad():
                    snaps1.append(sample(ema.shadow, eps_snap, 1, "chain").cpu().numpy())
                    snaps2.append(sample(ema.shadow, eps_snap, 2, sampler, noise_seed=300).cpu().numpy())
                    snaps4.append(sample(ema.shadow, eps_snap, 4, sampler, noise_seed=300).cpu().numpy())
                if demo:
                    # The raw student: the direction its own gradient is pushing the anchors right now.
                    c = construction(args.variant, student, fake, teacher, *anchor_set, drift=drift_vis)
                    for k in cons:
                        cons[k].append(c[k].cpu().numpy())
                snap_at.append(step)
            if step % cfg.quick_every == 0 or step == cfg.steps:
                q = quick_eval(step)
                qlog.add(**q)
                print(f"[{name} s{args.seed}] step {step:6d} sw2_k1 {q['sw2_k1']:.4f} "
                      f"sw2_k4 {q['sw2_k4']:.4f} coup {q['coupling_k1']:.4f}", flush=True)
            if step in milestones:
                (out / "milestones").mkdir(exist_ok=True)
                torch.save(dict(step=step, student=copy.deepcopy(student.state_dict()),
                                fake=copy.deepcopy(fake.state_dict()) if fake is not None else None),
                           out / "milestones" / f"step{step}.pt")
            sync(device)
            aside += time.time() - a0
        if step == cfg.steps:
            break

        lr = cosine_lr(step, cfg.steps, cfg.lr, cfg.lr_min, cfg.warmup)
        for grp in opt.param_groups:
            grp["lr"] = lr
        B = drift_n if kernel_neg else cfg.batch
        K = int(torch.randint(1, cfg.k_roll + 1, ()))

        # 1. rollout for the auxiliary network
        if renoise_dmd and not kernel_neg:
            for _ in range(cfg.n_fm):
                with torch.no_grad():
                    x_r, r = dmd_jump(B, K)
                fake_step(x_r, r)
        elif aux_fake:
            for _ in range(cfg.n_fm):
                fm_step(rollout(B, K))
        # 2. rollout for x_pred
        P = predict(B, K)
        if diag:
            xs = P.x_roll if P.x_roll is not None else untimed(rollout, B, K)
            untimed(lambda: [fm_step(xs) for _ in range(cfg.n_fm)])
        # 3. g
        g, W, gap, info = compute_g(P, step)
        # 4. loss for u_theta
        loss = pull_push(P.y, g, W)
        if not torch.isfinite(loss):
            diverged_at = step
            print(f"[{name} s{args.seed}] non-finite loss at step {step}; stopping", flush=True)
            break
        # 5. backward
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(student.parameters(), cfg.clip)
        opt.step()
        ema.update(student)
        n_mark += 1

        if step % cfg.log_every == 0:
            sync(device)
            now = time.time()
            ms = (now - t_mark - aside) / max(1, n_mark) * 1000
            ms_hist.append(ms)
            t_mark, n_mark, aside = now, 0, 0.0
            gap_c, d = gap.detach(), (P.t - P.r).squeeze(-1)
            row = dict(step=step, loss=loss.item(), gap=gap_c.mean().item(), grad_norm=gn.item(), lr=lr,
                       ms_per_step=ms, K=K)
            for lo, hi in [(0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.01)]:
                m = (d >= lo) & (d < hi)
                row[f"gap_d{lo:.2f}"] = gap_c[m].mean().item() if m.any() else float("nan")
            row.update(info)
            log.add(**row)

    log.save()
    qlog.save()
    save_net(out / "student_ema.pt", ema.shadow)
    save_net(out / "student.pt", student)
    if fake is not None:
        save_net(out / "fake.pt", fake)
    extra = {}
    if demo:
        extra = {f"c__{k}": np.stack(v) for k, v in cons.items()}
        extra["anchors"] = anchor_set[0].cpu().numpy()
        if drift_vis is not None:
            extra["drift_pos"] = drift_vis["pos"].cpu().numpy()
    np.savez_compressed(out / "snapshots.npz", steps=np.array(snap_at), k1=np.stack(snaps1),
                        k2=np.stack(snaps2), k4=np.stack(snaps4), **extra)
    ms_steady = ms_hist[len(ms_hist) // 5:] or ms_hist
    save_json(out / "config.json", dict(
        run=name, variant=args.variant, tag=args.tag, seed=args.seed, est=est, policy=policy,
        level=level, sampler=sampler, preset=preset.name, args=vars(args), student_cfg=asdict(cfg),
        cost=cost_per_sample(est, policy, args, cfg),
        ms_per_step=float(np.median(ms_steady)) if ms_steady else float("nan"),
        wall_s=time.time() - t_start, device=str(device), teacher_sha=teacher_sha(), diverged_at=diverged_at,
    ))
    print(f"[{name} s{args.seed}] done in {time.time() - t_start:.0f}s", flush=True)


# ====================================================================================================
# export_distribution: the six students of Section 6, their samples and KL over training, the final test KL
# ====================================================================================================
DEMO = [
    ("dmd-on-r0", "1  DMD, on-policy, r = 0"),
    ("mf-on-r0", "2  MeanFlow, on-policy, r = 0"),
    ("dmd-off-r0", "3  DMD, off-policy, r = 0"),
    ("mf-off-r0", "4  MeanFlow, off-policy, r = 0"),
    ("dmd-on-all", "5  DMD, on-policy, r random"),
    ("mf-on-all", "6  MeanFlow, on-policy, r random"),
]
VIEW = 3.2
KS = (1, 2, 4)
C = dict(data="#4C72B0", teacher="#2CA02C", student="#FF7F0E", neg_mf="#D62728", neg_dmd="#9467BD",
         query="#7F7F7F", k4="#8C564B")


def kl_pair(x, ref, n):
    x, ref = torch.as_tensor(x[:n]), ref[:n]
    return knn_kl(x, ref), knn_kl(ref, x)


@torch.no_grad()
def final_kl(root, refs, device, n):
    """Test-time KL of every seed's EMA student at K = 1, 2, 4 against held-out data (ref_A)."""
    A, B = refs["A"][:n], refs["B"][:n]
    floor = dict(kl_rev=knn_kl(B, A), kl_fwd=knn_kl(A, B), sw2=sw2(B, A, 500))
    table = CSVLog(Path(root).parent / "final_kl.csv")
    eps = cpu_randn((n, DIM), 4242).to(device)
    for run, _ in DEMO:
        for d in sorted(Path(root, run).glob("seed*")):
            student = load_net(d / "student_ema.pt", device)
            for K in KS:
                samp = "chain" if K == 1 else native_sampler(load_json(d / "config.json")["level"])
                x = sample(student, eps, K, samp, noise_seed=4300)
                table.add(run=run, seed=int(d.name[4:]), K=K, sampler=samp, kl_rev=knn_kl(x, A),
                          kl_fwd=knn_kl(A, x), sw2=sw2(x, A, 500))
    table.save()
    return table.rows, floor


def export_distribution(root, preset_name, device):
    preset = get_preset(preset_name)
    data = make_data(preset.data.noise, device)
    refs = load_refs(data, preset.data.n_ref)
    n_kl = 2000
    arrays = dict(data=refs["A"][:2000].numpy(), view=np.float32(VIEW))
    if DIM == 3:
        arrays.update(edges=data.edges(), view=np.float32(1.05 * refs["A"].abs().max().item()))
    meta = dict(runs=[r for r, _ in DEMO], labels=dict(DEMO))
    for run, _ in DEMO:
        seeds = sorted(Path(root, run).glob("seed*"))
        snaps = [np.load(d / "snapshots.npz") for d in seeds]
        s0 = snaps[0]
        arrays.update({f"{run}__snap_steps": s0["steps"], f"{run}__k1": s0["k1"], f"{run}__k2": s0["k2"],
                       f"{run}__k4": s0["k4"], f"{run}__anchors": s0["anchors"]})
        arrays.update({f"{run}__c__{k}": s0[f"c__{k}"] for k in FIELDS})
        # KL over training from each seed's snapshot clouds (reverse = KL(student || data)).
        for K in (1, 2):
            kl = np.array([[kl_pair(cloud, refs["A"], n_kl) for cloud in s[f"k{K}"]] for s in snaps])
            arrays[f"{run}__kl_rev_k{K}"], arrays[f"{run}__kl_fwd_k{K}"] = kl[..., 0], kl[..., 1]
        qs = [read_csv(d / "quick_log.csv") for d in seeds]
        arrays[f"{run}__q_step"] = np.array([row["step"] for row in qs[0]], dtype=np.float32)
        for k in ("sw2_k1", "sw2_k4", "coupling_k1"):
            arrays[f"{run}__{k}"] = np.array([[row[k] for row in q] for q in qs], dtype=np.float32)
        print(f"[demo] {run}: {len(s0['steps'])} snapshots, {len(seeds)} seeds", flush=True)
    rows, floor = final_kl(root, refs, device, preset.data.n_ref)
    meta["final_kl_floor"] = floor
    for run, _ in DEMO:
        for K in KS:
            sel = [r for r in rows if r["run"] == run and r["K"] == K]
            for k in ("kl_rev", "kl_fwd", "sw2"):
                arrays[f"{run}__final_{k}_k{K}"] = np.array([r[k] for r in sel], dtype=np.float32)
    out = ART / "assets" / "distribution_demo.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, meta=np.array(json.dumps(meta)), **arrays)
    print(f"[demo] {out}", flush=True)
    print_final_distribution(out)
    return out


def print_final_distribution(path):
    d = np.load(path)
    meta = json.loads(str(d["meta"]))
    f = meta["final_kl_floor"]
    print(f"\nFinal test KL vs held-out data (nats, kNN k=5; mean ± sd over seeds). Data floor: "
          f"KL(rev) {f['kl_rev']:.3f}, KL(fwd) {f['kl_fwd']:.3f}, SW2 {f['sw2']:.3f}")
    print(f"{'setup':34s} " + "  ".join(f"{'K=' + str(K) + ' rev':>12s} {'fwd':>12s}" for K in KS))
    for run in meta["runs"]:
        cells = []
        for K in KS:
            for k in ("kl_rev", "kl_fwd"):
                v = d[f"{run}__final_{k}_k{K}"]
                cells.append(f"{v.mean():6.3f}±{v.std():.3f}")
        print(f"{meta['labels'][run]:34s} " + "  ".join(f"{cells[2 * i]:>12s} {cells[2 * i + 1]:>12s}"
                                                       for i in range(len(KS))))


def frame(ax, title=None):
    ax.set_xlim(-VIEW, VIEW)
    ax.set_ylim(-VIEW, VIEW)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=7)


def dots(ax, x, c, s=1.2, alpha=0.45):
    ax.scatter(x[:, 0], x[:, 1], s=s, c=c, alpha=alpha, lw=0)


# ====================================================================================================
# animations of Sections 2 and 6 (GIF and MP4)
# ====================================================================================================
OUT = MATERIALS
STYLE = {"dmd-on-r0": (C["neg_dmd"], "-"), "dmd-off-r0": (C["neg_dmd"], "--"), "dmd-on-all": (C["neg_dmd"], ":"),
         "mf-on-r0": (C["student"], "-"), "mf-off-r0": (C["student"], "--"), "mf-on-all": (C["student"], ":")}


def grab(fig):
    fig.canvas.draw()
    return np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()


def write(frames, name, fps, hold):
    frames = frames + [frames[-1]] * hold
    OUT.mkdir(parents=True, exist_ok=True)
    imgs = [Image.fromarray(f) for f in frames]
    imgs[0].save(OUT / f"{name}.gif", save_all=True, append_images=imgs[1:], duration=int(1000 / fps), loop=0,
                 optimize=True)
    h, w = (s - s % 2 for s in frames[0].shape[:2])
    with av.open(str(OUT / f"{name}.mp4"), "w") as box:
        stream = box.add_stream("libx264", rate=fps)
        stream.width, stream.height, stream.pix_fmt = w, h, "yuv420p"
        for f in frames:
            for packet in stream.encode(av.VideoFrame.from_ndarray(f[:h, :w], format="rgb24")):
                box.mux(packet)
        for packet in stream.encode():
            box.mux(packet)
    print(f"[anim] {OUT / name}.gif / .mp4 ({len(frames)} frames)", flush=True)


PULL_PUSH = [
    ("mf-on-all", "MeanFlow",
     "$x_{\\mathrm{pred}} = X[x_t, u_\\theta, r]$: the student's prediction from $x_t$\n"
     "$x_{\\mathrm{pos}}, x_{\\mathrm{neg}}$: tangent points from the same $x_t$"),
    ("dmd-on-all", "DMD",
     "$x_{\\mathrm{pred}}$: the student's own point at level $r$ (its sample at $r=0$)\n"
     "$x_{\\mathrm{pos}}, x_{\\mathrm{neg}}$: teacher's and fake's estimates at renoised $x_{t'}$"),
]


def farthest_points(x, k, eligible):
    """Greedy farthest-point choice of k well-separated rows of x among the eligible ones."""
    idx = [int(np.flatnonzero(eligible)[np.argmax(np.linalg.norm(x[eligible], axis=1))])]
    while len(idx) < k:
        dist = np.min(np.linalg.norm(x[:, None] - x[idx][None], axis=-1), axis=1)
        dist[~eligible] = -1
        dist[idx] = -1
        idx.append(int(np.argmax(dist)))
    return idx


def _arrow(ax, p, q, c, lw, z=4):
    ax.annotate("", xy=q, xytext=p, zorder=z,
                arrowprops=dict(arrowstyle="-|>", color=c, lw=lw, shrinkA=0, shrinkB=3, mutation_scale=13))


def anim_directions(d, meta, fps, hold, k=3):
    """Attraction toward x_pos, repulsion from x_neg and their sum on k fixed anchors, over training."""
    steps = d[f"{PULL_PUSH[0][0]}__snap_steps"]
    pos_c, neg_c = C["teacher"], C["neg_mf"]
    setup = []
    for run, name, explain in PULL_PUSH:
        xp, pos, neg = (d[f"{run}__c__{f}"] for f in ("x_pred", "x_pos", "x_neg"))   # [frames, anchors, 2]
        size = np.linalg.norm(pos - neg, axis=-1).mean(0)
        chosen = farthest_points(xp[-1], k, size >= np.quantile(size, 0.3))
        upd_size = np.linalg.norm((pos - neg)[1:, chosen], axis=-1)
        scale = max(1.0, round(0.5 / np.median(upd_size)))
        pts = np.concatenate([a[:, chosen].reshape(-1, 2) for a in
                              (xp, pos, neg, 2 * xp - neg, xp + scale * (pos - neg))])
        lo, hi = np.quantile(pts, 0.02, axis=0), np.quantile(pts, 0.98, axis=0)
        c, half = (lo + hi) / 2, max(1.3, (hi - lo).max() / 2 + 0.3)
        setup.append((run, name, explain, xp, pos, neg, chosen, scale, c, half))
    fig, axs = plt.subplots(1, 2, figsize=(12.5, 6.8), layout="constrained")
    data = d["data"]
    frames = []
    for p in range(len(steps)):
        for ax, (run, name, explain, xp, pos, neg, chosen, scale, c, half) in zip(axs, setup):
            ax.cla()
            ax.scatter(data[:, 0], data[:, 1], s=1.2, c="0.85", lw=0, zorder=0)
            for i in chosen:
                ax.plot(*np.stack([xp[p, i], pos[p, i]]).T, c=pos_c, lw=0.7, ls=":", zorder=1)
                ax.plot(*np.stack([xp[p, i], neg[p, i]]).T, c=neg_c, lw=0.7, ls=":", zorder=1)
                _arrow(ax, xp[p, i], pos[p, i], pos_c, 2.0)
                _arrow(ax, xp[p, i], 2 * xp[p, i] - neg[p, i], neg_c, 2.0)
                _arrow(ax, xp[p, i], xp[p, i] + scale * (pos[p, i] - neg[p, i]), "k", 2.6, z=6)
                ax.scatter(*pos[p, i], c=pos_c, s=55, zorder=5, edgecolors="white", linewidths=0.8)
                ax.scatter(*neg[p, i], c=neg_c, s=55, zorder=5, edgecolors="white", linewidths=0.8)
                ax.scatter(*xp[p, i], c="k", s=65, zorder=7, edgecolors="white", linewidths=0.8)
            ax.set_xlim(c[0] - half * 1.15, c[0] + half * 1.15)
            ax.set_ylim(c[1] - half, c[1] + half * 1.3)
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])
            for s in ax.spines.values():
                s.set_color("0.8")
            ax.set_title(f"{name}  ($r = 0.3$)", fontsize=13)
            ax.text(0.02, 0.02, explain, transform=ax.transAxes, fontsize=9.5, va="bottom",
                    bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="0.8"))
            upd = "update $x_{\\mathrm{pos}} - x_{\\mathrm{neg}}$" + (f"  (drawn $\\times${scale:g})" if scale > 1 else "")
            ax.legend(handles=[
                plt.Line2D([], [], marker="o", ls="", c="k", label="$x_{\\mathrm{pred}}$ (moved)"),
                plt.Line2D([], [], marker="o", ls="", c=pos_c, label="$x_{\\mathrm{pos}}$"),
                plt.Line2D([], [], marker="o", ls="", c=neg_c, label="$x_{\\mathrm{neg}}$"),
                plt.Line2D([], [], c=pos_c, lw=2, label="attraction  $x_{\\mathrm{pos}} - x_{\\mathrm{pred}}$"),
                plt.Line2D([], [], c=neg_c, lw=2, label="repulsion  $x_{\\mathrm{pred}} - x_{\\mathrm{neg}}$"),
                plt.Line2D([], [], c="k", lw=2.6, label=upd),
            ], loc="upper left", fontsize=9, framealpha=0.9, edgecolor="0.85")
        fig.suptitle(f"Attraction, repulsion and the update on three fixed points during training   "
                     f"step {int(steps[p])}", fontsize=12)
        frames.append(grab(fig))
    plt.close(fig)
    write(frames, "directions", fps, hold)


def frame3d(ax, view, edges, azim):
    """3D panel: the filled cubes of the target as wireframes, a fixed box and a slowly turning camera."""
    from mpl_toolkits.mplot3d.art3d import Line3DCollection
    ax.add_collection3d(Line3DCollection(edges, colors="0.55", linewidths=0.5, alpha=0.6))
    for set_lim in (ax.set_xlim, ax.set_ylim, ax.set_zlim):
        set_lim(-view, view)
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=22, azim=azim)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_pane_color((1, 1, 1, 0))
        axis.line.set_color("0.85")


def anim_samples(d, meta, K, fps, hold, out_name=None):
    runs, labels = meta["runs"], meta["labels"]
    steps = d[f"{runs[0]}__snap_steps"]
    floor = meta["final_kl_floor"]
    three_d = d["data"].shape[1] == 3
    if three_d:
        fig = plt.figure(figsize=(16, 8.4), layout="constrained")
        cloud_axes = [fig.add_subplot(2, 4, i + 1, projection="3d") for i in range(6)]
        ax_rev, ax_fwd = fig.add_subplot(2, 4, 7), fig.add_subplot(2, 4, 8)
    else:
        fig, axs = plt.subplots(2, 4, figsize=(16, 8.4), layout="constrained")
        cloud_axes, (ax_rev, ax_fwd) = list(axs.flat[:6]), axs.flat[6:]
    frames = []
    for p in range(len(steps)):
        for ax, run in zip(cloud_axes, runs):
            ax.cla()
            cloud = d[f"{run}__k{K}"][p]
            if three_d:
                # One full turn of the camera over the animation, so the depth of the cubes can be read.
                frame3d(ax, float(d["view"]), d["edges"], azim=-60 + 360 * p / len(steps))
                ax.scatter(*cloud.T, s=min(3.0, 6000 / len(cloud)), c=STYLE[run][0], alpha=0.6, lw=0,
                           depthshade=False)
            else:
                frame(ax)
                dots(ax, d["data"], C["data"], s=1.5, alpha=0.18)
                dots(ax, cloud, STYLE[run][0], s=2.0, alpha=0.55)
            kl = d[f"{run}__kl_rev_k{K}"][0, p]
            ax.set_title(f"{labels[run]}\nKL(student || data) = {kl:.3f}", fontsize=9)
        for ax, key, name, fl in ((ax_rev, "rev", "KL(student || data)  [reverse]", floor["kl_rev"]),
                                  (ax_fwd, "fwd", "KL(data || student)  [forward]", floor["kl_fwd"])):
            ax.cla()
            for run in runs:
                c, ls = STYLE[run]
                ax.plot(steps, d[f"{run}__kl_{key}_k{K}"].mean(0), c=c, ls=ls, lw=1.6, label=labels[run][3:])
            ax.axhline(fl, c="k", lw=0.8, ls=(0, (1, 2)), label="data vs data floor")
            ax.axvline(steps[p], c="0.3", lw=1)
            ax.set_yscale("symlog", linthresh=0.05)
            ax.set_ylim(bottom=-0.01)
            ax.set(xlabel="student step", ylabel="nats (kNN, k=5; mean over 3 seeds)", title=name)
            ax.title.set_fontsize(9)
            ax.tick_params(labelsize=7)
        ax_rev.legend(fontsize=7, loc="upper right")
        target = "the 14 cubes of the 3D checkerboard as wireframes" if three_d else "data in blue"
        fig.suptitle(f"{K}-step samples over training ({target})   step {int(steps[p])}", fontsize=12)
        frames.append(grab(fig))
    plt.close(fig)
    write(frames, out_name or f"samples_k{K}", fps, hold)


def anim_single(root, run, name, title, preset_name, fps, hold, n_kl=2000):
    """One method: 1-step and 2-step samples over training, and its KL to the data (mean over seeds)."""
    refs = load_refs(make_data(get_preset(preset_name).data.noise, "cpu"), get_preset(preset_name).data.n_ref)
    A, B = refs["A"][:n_kl], refs["B"][:n_kl]
    floor = knn_kl(B, A)
    snaps = [np.load(d / "snapshots.npz") for d in sorted(Path(root, run).glob("seed*"))]
    steps = snaps[0]["steps"]
    kl = {K: np.array([[[knn_kl(torch.as_tensor(c[:n_kl]), A), knn_kl(A, torch.as_tensor(c[:n_kl]))]
                        for c in s[f"k{K}"]] for s in snaps]) for K in (1, 2)}
    fig, axs = plt.subplots(1, 3, figsize=(15, 5.2), layout="constrained",
                            gridspec_kw=dict(width_ratios=[1, 1, 1.35]))
    frames = []
    for p in range(len(steps)):
        for ax, K, c in ((axs[0], 1, C["student"]), (axs[1], 2, C["k4"])):
            ax.cla()
            frame(ax)
            dots(ax, A.numpy(), C["data"], s=1.5, alpha=0.2)
            dots(ax, snaps[0][f"k{K}"][p], c, s=2.2, alpha=0.6)
            ax.set_title(f"{K}-step samples   KL(student || data) = {kl[K][0, p, 0]:.3f}", fontsize=10)
        ax = axs[2]
        ax.cla()
        for K, c in ((1, C["student"]), (2, C["k4"])):
            ax.plot(steps, kl[K][..., 0].mean(0), c=c, lw=1.8, label=f"KL(student || data), {K} step{'s' * (K > 1)}")
            ax.plot(steps, kl[K][..., 1].mean(0), c=c, lw=1.2, ls="--",
                    label=f"KL(data || student), {K} step{'s' * (K > 1)}")
        ax.axhline(floor, c="k", lw=0.8, ls=(0, (1, 2)), label="data vs data floor")
        ax.axvline(steps[p], c="0.3", lw=1)
        ax.set_yscale("symlog", linthresh=0.05)
        ax.set_ylim(bottom=-0.01)
        ax.set(xlabel="student step", ylabel=f"nats (kNN, k=5; mean over {len(snaps)} seeds)")
        ax.legend(fontsize=8, loc="upper right")
        fig.suptitle(f"{title}   step {int(steps[p])}   (data in blue)", fontsize=12)
        frames.append(grab(fig))
    plt.close(fig)
    write(frames, name, fps, hold)


# ====================================================================================================
# Section 1: the teacher's ODE with K Euler steps, and the flow map on one trajectory
# ====================================================================================================
STEP_KS = tuple(range(1, 65))


@torch.no_grad()
def teacher_steps(preset_name="smoke", device="cpu", n=4000):
    """K-step Euler samples of the teacher for K = 1..64 and a fine RK4 solve from the same noises, with their
    SW2 and KL(samples || data) to held-out data, and the data-against-data floor."""
    preset, dev = get_preset(preset_name), pick_device(device)
    teacher = load_teacher(dev)
    refs = load_refs(make_data(preset.data.noise, dev), preset.data.n_ref)
    A, B = refs["A"][:n], refs["B"][:n]
    eps = cpu_randn((n, DIM), 4343).to(dev)
    ref = teacher.rk4(eps, 100)
    clouds, sws, kls = [], [], []
    for K in STEP_KS:
        x = teacher.euler(eps, K)
        clouds.append(x.cpu().numpy())
        sws.append(sw2(x, A, 500))
        kls.append(knn_kl(x, A))
    return dict(ks=np.array(STEP_KS), clouds=np.stack(clouds), sw2=np.array(sws), kl=np.array(kls),
                ref=ref.cpu().numpy(), ref_sw2=sw2(ref, A, 500), ref_kl=knn_kl(ref, A),
                floor_sw2=sw2(B, A, 500), floor_kl=knn_kl(B, A), data=A.cpu().numpy())


def anim_teacher_steps(d, fps, hold, n_show=2000):
    """K-step Euler samples of the teacher next to the fine ODE solve, and the two metrics against K."""
    fig, axs = plt.subplots(1, 4, figsize=(17, 4.7), layout="constrained",
                            gridspec_kw=dict(width_ratios=[1, 1, 1.2, 1.2]))
    frames = []
    for p, K in enumerate(d["ks"]):
        for ax, x, title in ((axs[0], d["clouds"][p], f"{K}-step Euler sampling"),
                             (axs[1], d["ref"], "fine ODE solve (RK4, 100 steps)")):
            ax.cla()
            frame(ax)
            dots(ax, d["data"][:n_show], C["data"], s=1.5, alpha=0.18)
            dots(ax, x[:n_show], C["teacher"], s=2.0, alpha=0.55)
            ax.set_title(title, fontsize=11)
        for ax, key, name in ((axs[2], "sw2", "SW2 to the data"), (axs[3], "kl", "KL(samples || data)")):
            ax.cla()
            ax.plot(d["ks"], d[key], c=C["teacher"], lw=1.8, label="K-step Euler")
            ax.scatter([K], [d[key][p]], c=C["teacher"], s=40, zorder=5)
            ax.axhline(d[f"ref_{key}"], c=C["teacher"], lw=1, ls="--", label="fine ODE solve")
            ax.axhline(d[f"floor_{key}"], c="k", lw=0.8, ls=(0, (1, 2)), label="data vs data floor")
            ax.set_xscale("log", base=2)
            ax.set_yscale("symlog", linthresh=0.05)
            ax.set_ylim(bottom=-0.01)
            ax.set(xlabel="number of steps K", title=f"{name} = {d[key][p]:.3f}")
            ax.title.set_fontsize(11)
            ax.legend(fontsize=8, loc="upper right")
        fig.suptitle(f"The teacher's ODE from noise to data with K Euler steps (data in blue)   K = {K}", fontsize=12)
        frames.append(grab(fig))
    plt.close(fig)
    write(frames, "teacher_steps", fps, hold)


def anim_flow_map(fps, hold, n_ode=48, n_jump=20, n_tan=16, n_zero=16, pause=6):
    """The schematic of f2_tangent_vs_chord (a), animated: ODE sampling along the trajectory with the tangent at
    every step, then one jump along the secant (average velocity, flow map), then one Euler step along the
    tangent at x_t (tangent point), then all three continued to level 0 with the distributions they produce."""
    with matplotlib.rc_context(FIG_RC):
        _anim_flow_map(fps, hold, n_ode, n_jump, n_tan, n_zero, pause)


def _anim_flow_map(fps, hold, n_ode, n_jump, n_tan, n_zero, pause):
    x_t = np.array([0.0, 0.0])
    x_r = np.array([S_R, teacher_curve(S_R)])
    x_tan = np.array([S_R, TEACHER_SLOPE * S_R])
    slope = lambda s: TEACHER_SLOPE - 2 * 1.25 * s
    fig = plt.figure(figsize=(7.2, 4.9))
    ax = fig.add_axes([0.0, 0.0, 1.0, 0.86])
    title_text = fig.text(0.5, 0.97, "", ha="center", va="top", fontsize=12.5, linespacing=1.4)

    def base(title):
        ax.cla()
        level_axis(ax)
        ax.set_ylim(-0.42, 1.12)
        gradient_curve(ax, teacher_curve, 1.0, alpha=0.3)
        dot(ax, x_t, "k")
        ax.text(x_t[0] + 0.01, x_t[1] - 0.09, "$x_t$", fontsize=13, va="top")
        ax.add_line(GradientLegendLine("teacher ODE trajectory"))
        ax.legend(loc="upper left", bbox_to_anchor=(0.0, 1.0), frameon=False, fontsize=9.5)
        title_text.set_text(title)

    def flow_map_drawn():
        gradient_curve(ax, teacher_curve, S_R)
        arrow(ax, x_t, x_r, "k", lw=2.2)
        dot(ax, x_r, "k")
        ax.text(x_r[0] + 0.03, x_r[1] - 0.13, "flow map $x_r$\n$X[x_t, u, r]$", ha="left", fontsize=11)

    frames = []
    for s in np.linspace(0.0, S_R, n_ode):
        base("ODE sampling follows the trajectory in many small steps,\n"
             "each along the tangent $v(x_s, s)$")
        if s > 0:
            gradient_curve(ax, teacher_curve, S_R, s_stop=s)
        d = np.array([1.0, slope(s)])
        p = np.array([s, teacher_curve(s)])
        arrow(ax, p, p + 0.22 * d / np.linalg.norm(d), FC["grey"], lw=2.0, z=6)
        dot(ax, p, "k", z=7)
        ax.text(p[0], p[1] + 0.07, "$v(x_s, s)$", color="0.35", fontsize=11, ha="right")
        frames.append(grab(fig))
    frames += [frames[-1]] * pause

    title_jump = "the average velocity $u(x_t, t, r)$ jumps along the secant\nin one step and lands on the flow map $x_r$"
    for a in np.linspace(0.05, 1.0, n_jump):
        base(title_jump)
        gradient_curve(ax, teacher_curve, S_R)
        p = x_t + a * (x_r - x_t)
        arrow(ax, x_t, p, "k", lw=2.2)
        dot(ax, p, "k", z=7)
        if a >= 1:
            ax.text(x_r[0] + 0.03, x_r[1] - 0.13, "flow map $x_r$\n$X[x_t, u, r]$", ha="left", fontsize=11)
        frames.append(grab(fig))
    frames += [frames[-1]] * pause

    for a in np.linspace(0.05, 1.0, n_tan):
        base("one Euler step along the tangent at $x_t$ lands at the\ntangent point, off the trajectory")
        flow_map_drawn()
        q = x_t + a * (x_tan - x_t)
        arrow(ax, x_t, q, FC["grey"], ls="--", lw=2.0)
        dot(ax, q, FC["grey"])
        if a >= 1:
            ax.annotate("", xy=(S_R + 0.035, x_tan[1]), xytext=(S_R + 0.035, x_r[1]),
                        arrowprops=dict(arrowstyle="<->", color=FC["neg"], lw=1.3))
            ax.text(S_R + 0.06, (x_tan[1] + x_r[1]) / 2, "curvature\ngap", color=FC["neg"], fontsize=10,
                    va="center")
            ax.text(x_tan[0] + 0.03, x_tan[1] + 0.02, "tangent point\n$X[x_t, v, r]$", ha="left", va="bottom",
                    fontsize=11, color="0.25")
        frames.append(grab(fig))
    frames += [frames[-1]] * pause

    # Continued to level 0: the trajectory and the secant end at x_0, one Euler step with v lands far away.
    # Outputs from states spread around x_t give a peak at each endpoint, and the two peaks do not overlap.
    x_0 = np.array([1.0, teacher_curve(1.0)])
    x_eul = np.array([1.0, TEACHER_SLOPE])
    sigma, amp = 0.09, 0.3
    ys = np.linspace(-4, 4, 200) * sigma
    bump = np.exp(-0.5 * (ys / sigma) ** 2)
    title_zero = ("continued to level 0, the trajectory and the secant both end at $x_0$,\n"
                  "while one Euler step with $v$ lands far away")

    def zero_frame(a_arrow, zoom, a_dist):
        base(title_zero)
        ax.set_xlim(-0.12, 1.15 + 0.2 * zoom)
        ax.set_ylim(-0.42, 1.12 + 0.8 * zoom)
        gradient_curve(ax, teacher_curve, 1.0)
        arrow(ax, x_t, x_t + a_arrow * (x_0 - x_t), "k", lw=2.2)
        arrow(ax, x_t, x_t + a_arrow * (x_eul - x_t), FC["grey"], ls="--", lw=2.0)
        dot(ax, x_tan, FC["grey"], s=30)
        if a_dist > 0:
            for centre, c in ((x_0[1], "#4C72B0"), (x_eul[1], FC["grey"])):
                ax.fill_betweenx(centre + ys, 1 - a_dist * amp * bump, 1, color=c, alpha=0.35, lw=0, zorder=3)
                ax.plot(1 - a_dist * amp * bump, centre + ys, c=c, lw=1.2, zorder=3)
        dot(ax, x_t + a_arrow * (x_0 - x_t), "k", z=7)
        dot(ax, x_t + a_arrow * (x_eul - x_t), FC["grey"], z=7)
        if a_arrow >= 1:
            ax.text(1.03, x_0[1] - 0.02, "$x_0 = X[x_t, u, 0]$", fontsize=11, va="center")
            ax.text(1.03, x_eul[1] + 0.02, "$X[x_t, v, 0]$", fontsize=11, va="center", color="0.25")
        if a_dist >= 1:
            ax.text(1 - amp - 0.02, x_0[1], "along the trajectory\nor the secant", ha="right", va="center",
                    fontsize=10, color="#4C72B0")
            ax.text(1 - amp - 0.02, x_eul[1], "one Euler step\nwith $v$", ha="right", va="center", fontsize=10,
                    color="0.35")
            ax.annotate("", xy=(1.06, x_eul[1] - 0.12), xytext=(1.06, x_0[1] + 0.12),
                        arrowprops=dict(arrowstyle="<->", color=FC["neg"], lw=1.3))
            ax.text(1.09, (x_0[1] + x_eul[1]) / 2, "the two\ndistributions\ndo not match", color=FC["neg"],
                    fontsize=10, va="center")
        return grab(fig)

    frames += [zero_frame(a, a, 0.0) for a in np.linspace(0.05, 1.0, n_zero)]
    frames += [zero_frame(1.0, 1.0, a) for a in np.linspace(0.1, 1.0, n_zero // 2)]
    plt.close(fig)
    write(frames, "flow_map", fps, hold)


# ====================================================================================================
# Section 8: pos / neg from N samples (native drift, FM-ideal drift, semi-drift): exports, figure, animations
# ====================================================================================================
NS = (2, 4, 8, 16, 64, 256, 1024)
DIRECTION_NS = (256, 8, 2)
FAMILIES = {
    "drift": dict(run=lambda n: f"drift-on-r0+n{n}",
                  title="Native drift: Laplacian kernel",
                  xlabel="N: generated (negatives) and data (positives) samples per step"),
    "drift_gauss": dict(run=lambda n: f"drift-on-r0+g{n}",
                        title="FM-ideal drift: Gaussian kernel",
                        xlabel="N: generated (negatives) and data (positives) samples per step"),
    "kneg": dict(run=lambda n: f"dmd-on-r0+kneg{n}",
                 title="Semi-drift: teacher positive, Gaussian-kernel negative",
                 xlabel="N: generated samples per step (negatives)"),
}
default_root = lambda family: ART / family / "runs"


@torch.no_grad()
def export_kernel(family, root, preset_name, device):
    run_of = FAMILIES[family]["run"]
    preset = get_preset(preset_name)
    refs = load_refs(make_data(preset.data.noise, device), preset.data.n_ref)
    A, B = refs["A"], refs["B"]
    n_kl, n = 2000, preset.data.n_ref
    arrays = dict(data=A[:2000].numpy())
    floor = dict(kl_rev=knn_kl(B[:n], A[:n]), kl_fwd=knn_kl(A[:n], B[:n]))
    table = CSVLog(Path(root).parent / "final_kl.csv")
    eps = cpu_randn((n, DIM), 4242).to(device)
    for N in NS:
        run = run_of(N)
        seeds = sorted(Path(root, run).glob("seed*"))
        snaps = [np.load(d / "snapshots.npz") for d in seeds]
        s0 = snaps[0]
        arrays.update({f"{N}__{k}": s0[k] for k in ("steps", "k1", "k2", "drift_pos") if k in s0})
        arrays.update({f"{N}__c__{k}": s0[f"c__{k}"] for k in ("x_pred", "x_pos", "x_neg", "y_neg")
                       if f"c__{k}" in s0})
        for K in (1, 2):
            kl = np.array([[kl_pair(cloud, A, n_kl) for cloud in s[f"k{K}"]] for s in snaps])
            arrays[f"{N}__kl_rev_k{K}"], arrays[f"{N}__kl_fwd_k{K}"] = kl[..., 0], kl[..., 1]
        qs = [read_csv(d / "quick_log.csv") for d in seeds]
        arrays[f"{N}__sw2_k1"] = np.array([[row["sw2_k1"] for row in q] for q in qs], dtype=np.float32)
        for d in seeds:
            student = load_net(d / "student_ema.pt", device)
            for K in KS:
                samp = "chain" if K == 1 else "renoise"
                x = sample(student, eps, K, samp, noise_seed=4300)
                table.add(run=run, n=N, seed=int(d.name[4:]), K=K, sampler=samp, kl_rev=knn_kl(x, A[:n]),
                          kl_fwd=knn_kl(A[:n], x), sw2=sw2(x, A[:n], 500))
        print(f"[{family}] N={N}: {len(seeds)} seeds", flush=True)
    table.save()
    for N in NS:
        for K in KS:
            sel = [r for r in table.rows if r["n"] == N and r["K"] == K]
            for k in ("kl_rev", "kl_fwd", "sw2"):
                arrays[f"{N}__final_{k}_k{K}"] = np.array([r[k] for r in sel], dtype=np.float32)
    out = ART / "assets" / f"{family}_demo.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, meta=np.array(json.dumps(dict(family=family, ns=NS, floor=floor))), **arrays)
    print_final_kernel(out)
    return out


def print_final_kernel(path):
    d = np.load(path)
    f = json.loads(str(d["meta"]))["floor"]
    family = json.loads(str(d["meta"]))["family"]
    print(f"\n{family}: final test KL vs held-out data (nats, kNN k=5; mean ± sd over seeds). "
          f"Data floor: rev {f['kl_rev']:.3f}, fwd {f['kl_fwd']:.3f}")
    print(f"{'N':>6s} " + "  ".join(f"{'K=' + str(K) + ' rev':>13s} {'fwd':>13s} {'sw2':>13s}" for K in KS))
    for N in NS:
        cells = [f"{d[f'{N}__final_{k}_k{K}'].mean():7.3f}±{d[f'{N}__final_{k}_k{K}'].std():.3f}"
                 for K in KS for k in ("kl_rev", "kl_fwd", "sw2")]
        print(f"{N:6d} " + "  ".join(f"{c:>13s}" for c in cells))


def ref_kl():
    """Final one-step reverse KL of the on-policy r = 0 DMD and MeanFlow demo runs, if exported."""
    p = ART / "assets" / "distribution_demo.npz"
    if not p.exists():
        return {}
    d = np.load(p)
    return {name: d[f"{run}__final_kl_rev_k1"].mean() for run, name in
            (("dmd-on-r0", "DMD (on-policy, r = 0)"),
             ("mf-on-r0", "MeanFlow (on-policy, r = 0)"))}


def save(fig, name):
    MATERIALS.mkdir(exist_ok=True)
    for ext in ("png", "svg"):
        fig.savefig(MATERIALS / f"{name}.{ext}", dpi=150)
    plt.close(fig)
    print(f"[kl] {MATERIALS / name}.png", flush=True)


def kl_vs_n(d):
    fig, ax = plt.subplots(figsize=(6.2, 4.0), layout="constrained")
    draw_kl_vs_n(ax, d)
    save(fig, f"{json.loads(str(d['meta']))['family']}_kl_vs_n")


def kl_vs_n_pair(ds):
    """The families side by side on one y-axis."""
    fig, axs = plt.subplots(1, len(ds), figsize=(5.2 * len(ds), 4.3), layout="constrained", sharey=True)
    for i, (ax, d) in enumerate(zip(axs, ds)):
        draw_kl_vs_n(ax, d, legend=(i == len(ds) - 1), ylabel=(i == 0), letter="abcd"[i])
    save(fig, "kl_vs_n_" + "_".join(json.loads(str(d["meta"]))["family"] for d in ds))


def draw_kl_vs_n(ax, d, legend=True, ylabel=True, letter=None):
    meta = json.loads(str(d["meta"]))
    f, fam = meta["floor"], FAMILIES[meta["family"]]
    for K, c in ((1, C["student"]), (2, C["k4"])):
        for key, ls, name in (("kl_rev", "-", "KL(student || data)"), ("kl_fwd", "--", "KL(data || student)")):
            v = np.array([d[f"{N}__final_{key}_k{K}"] for N in NS])
            ax.errorbar(NS, v.mean(1), yerr=v.std(1), c=c, ls=ls, marker="o", ms=4, lw=1.5, capsize=3,
                        label=f"{name}, {K} step{'s' * (K > 1)}")
    for (name, v), c in zip(ref_kl().items(), (C["neg_dmd"], C["neg_mf"])):
        ax.axhline(v, c=c, lw=1, ls="-.", label=f"{name}, 1 step")
    ax.axhline(f["kl_rev"], c="k", lw=0.8, ls=(0, (1, 2)), label="data vs data floor")
    ax.set_xscale("log", base=2)
    ax.set_yscale("symlog", linthresh=0.1)
    ax.set_ylim(-0.01, 30)
    ax.set_xticks(NS, [str(N) for N in NS])
    ax.set(xlabel=fam["xlabel"], ylabel="nats (kNN, k=5; mean ± sd over 3 seeds)" if ylabel else None,
           title=(f"({letter}) " if letter else "") + f"{fam['title']}:\nfinal KL to the data vs N")
    ax.title.set_fontsize(10)
    ax.yaxis.set_tick_params(labelleft=True)
    if legend:
        ax.legend(fontsize=7, loc="upper right")


def anim_drift_directions(d, fps, hold, k=3):
    """x_pred, the kernel means x_pos / x_neg and the update on k fixed points, for a few N."""
    steps = d[f"{NS[0]}__steps"]
    pos_c, neg_c = C["teacher"], C["neg_mf"]
    xp0 = d[f"{DIRECTION_NS[0]}__c__x_pred"][0]
    chosen = farthest_points(xp0, k, np.ones(len(xp0), bool))
    ref = DIRECTION_NS[0]
    gap = np.linalg.norm((d[f"{ref}__c__x_pos"] - d[f"{ref}__c__x_neg"])[1:, chosen], axis=-1)
    scale = max(1.0, round(0.4 / np.median(gap)))
    fig, axs = plt.subplots(1, len(DIRECTION_NS), figsize=(5.6 * len(DIRECTION_NS), 6.4), layout="constrained")
    data = d["data"]
    frames = []
    for p in range(len(steps)):
        for ax, N in zip(axs, DIRECTION_NS):
            xp, pos, neg, y_neg = (d[f"{N}__c__{f}"][p] for f in ("x_pred", "x_pos", "x_neg", "y_neg"))
            y_pos = d[f"{N}__drift_pos"]
            ax.cla()
            ax.scatter(data[:, 0], data[:, 1], s=1.2, c="0.88", lw=0, zorder=0)
            big = N > 64
            ax.scatter(*y_pos.T, s=4 if big else 40, c=C["data"], alpha=0.35 if big else 0.9, lw=0, zorder=1,
                       marker="o")
            ax.scatter(*y_neg.T, s=4 if big else 40, c=C["student"], alpha=0.35 if big else 0.9, lw=0, zorder=1,
                       marker="o")
            for i in chosen:
                _arrow(ax, xp[i], pos[i], pos_c, 2.0)
                _arrow(ax, xp[i], 2 * xp[i] - neg[i], neg_c, 2.0)
                _arrow(ax, xp[i], xp[i] + scale * (pos[i] - neg[i]), "k", 2.6, z=6)
                ax.scatter(*pos[i], c=pos_c, s=55, zorder=5, edgecolors="white", linewidths=0.8)
                ax.scatter(*neg[i], c=neg_c, s=55, zorder=5, edgecolors="white", linewidths=0.8)
                ax.scatter(*xp[i], c="k", s=65, zorder=7, edgecolors="white", linewidths=0.8)
            shown = np.concatenate([xp[chosen], pos[chosen], neg[chosen], y_neg])
            half = max(2.6, 1.1 * np.abs(shown).max())
            ax.set_xlim(-half, half)
            ax.set_ylim(-half, half)
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])
            for s in ax.spines.values():
                s.set_color("0.8")
            zoom = f"   (view zoomed out {half / 2.6:.0f}x)" if half > 2.7 else ""
            ax.set_title(f"N = {N}{zoom}", fontsize=13)
        upd = "update $x_{\\mathrm{pos}} - x_{\\mathrm{neg}}$" + (f" (drawn $\\times${scale:g})" if scale > 1 else "")
        axs[0].legend(handles=[
            plt.Line2D([], [], marker="o", ls="", c=C["data"], label="positives: N data samples"),
            plt.Line2D([], [], marker="o", ls="", c=C["student"], label="negatives: N generated samples"),
            plt.Line2D([], [], marker="o", ls="", c="k", label="$x_{\\mathrm{pred}}$: a one-step sample"),
            plt.Line2D([], [], marker="o", ls="", c=pos_c, label="$x_{\\mathrm{pos}}$: kernel mean of positives"),
            plt.Line2D([], [], marker="o", ls="", c=neg_c, label="$x_{\\mathrm{neg}}$: kernel mean of negatives"),
            plt.Line2D([], [], c="k", lw=2.6, label=upd),
        ], loc="upper left", fontsize=9, framealpha=0.9, edgecolor="0.85")
        fig.suptitle(f"Drifting model: positives, negatives and the update on three fixed points   "
                     f"step {int(steps[p])}", fontsize=13)
        frames.append(grab(fig))
    plt.close(fig)
    write(frames, "drift_directions", fps, hold)


def anim_kernel_family(d, fps, hold, K=1):
    meta = json.loads(str(d["meta"]))
    f, family = meta["floor"], meta["family"]
    steps = d[f"{NS[0]}__steps"]
    colors = [matplotlib.colors.to_hex(c) for c in plt.cm.viridis(np.linspace(0, 0.9, len(NS)))]
    fig, axs = plt.subplots(2, 4, figsize=(16, 8.4), layout="constrained")
    cloud_axes, ax_kl = list(axs.flat[:7]), axs.flat[7]
    frames = []
    for p in range(len(steps)):
        for ax, N, c in zip(cloud_axes, NS, colors):
            ax.cla()
            frame(ax)
            dots(ax, d["data"], C["data"], s=1.5, alpha=0.18)
            dots(ax, d[f"{N}__k{K}"][p], c, s=2.0, alpha=0.6)
            kl = d[f"{N}__kl_rev_k{K}"][0, p]
            ax.set_title(f"N = {N}\nKL(student || data) = {kl:.3f}", fontsize=9)
        ax_kl.cla()
        for N, c in zip(NS, colors):
            ax_kl.plot(steps, d[f"{N}__kl_rev_k{K}"].mean(0), c=c, lw=1.6, label=f"N = {N}")
        ax_kl.axhline(f["kl_rev"], c="k", lw=0.8, ls=(0, (1, 2)), label="data vs data floor")
        ax_kl.axvline(steps[p], c="0.3", lw=1)
        ax_kl.set_yscale("symlog", linthresh=0.05)
        ax_kl.set_ylim(bottom=-0.01)
        ax_kl.set(xlabel="student step", ylabel="nats (kNN, k=5; mean over 3 seeds)",
                  title="KL(student || data)  [reverse]")
        ax_kl.title.set_fontsize(9)
        ax_kl.tick_params(labelsize=7)
        ax_kl.legend(fontsize=7, loc="upper right", ncol=2)
        fig.suptitle(f"{FAMILIES[family]['title']}, {K}-step samples over training (data in blue): "
                     f"N samples per step   step {int(steps[p])}", fontsize=12)
        frames.append(grab(fig))
    plt.close(fig)
    write(frames, f"{family}_samples" + ("" if K == 1 else f"_k{K}"), fps, hold)


def anim_kernel_compare(ds, fps, hold, K=1):
    """One row per family (one-step samples for every N, then its reverse KL), all on one clock."""
    metas = [json.loads(str(d["meta"])) for d in ds]
    steps = ds[0][f"{NS[0]}__steps"]
    colors = [matplotlib.colors.to_hex(c) for c in plt.cm.viridis(np.linspace(0, 0.9, len(NS)))]
    fig, axs = plt.subplots(len(ds), len(NS) + 1, figsize=(2.1 * (len(NS) + 1) + 0.6, 2.35 * len(ds)),
                            layout="constrained", gridspec_kw=dict(width_ratios=[1] * len(NS) + [1.5]))
    frames = []
    for p in range(len(steps)):
        for row, (d, meta) in enumerate(zip(ds, metas)):
            for ax, N, c in zip(axs[row, :-1], NS, colors):
                ax.cla()
                frame(ax)
                dots(ax, d["data"], C["data"], s=0.8, alpha=0.18)
                dots(ax, d[f"{N}__k{K}"][p], c, s=1.0, alpha=0.6)
                ax.set_title(f"N = {N}   KL {d[f'{N}__kl_rev_k{K}'][0, p]:.2f}", fontsize=8)
            name, what = FAMILIES[meta["family"]]["title"].split(": ", 1)
            axs[row, 0].set_ylabel(name + ":\n" + "\n".join(textwrap.wrap(what, 22)), fontsize=8.5)
            ax = axs[row, -1]
            ax.cla()
            for N, c in zip(NS, colors):
                ax.plot(steps, d[f"{N}__kl_rev_k{K}"].mean(0), c=c, lw=1.3, label=f"N = {N}")
            ax.axhline(meta["floor"]["kl_rev"], c="k", lw=0.8, ls=(0, (1, 2)))
            ax.axvline(steps[p], c="0.3", lw=1)
            ax.set_yscale("symlog", linthresh=0.05)
            ax.set_ylim(-0.01, 30)
            ax.tick_params(labelsize=6)
            ax.set_title("KL(student || data), mean of 3 seeds", fontsize=8)
            if row == len(ds) - 1:
                ax.set_xlabel("student step", fontsize=7)
            if row == 0:
                ax.legend(fontsize=5.5, ncol=2, loc="upper right")
        fig.suptitle(f"Pos / neg from N samples: {K}-step samples over training (data in blue)   "
                     f"step {int(steps[p])}", fontsize=11)
        frames.append(grab(fig))
    plt.close(fig)
    write(frames, "kernel_samples" + ("" if K == 1 else f"_k{K}"), fps, hold)


# ====================================================================================================
# static figures of Section 4
# ====================================================================================================
FIG_RC = {"mathtext.fontset": "cm", "font.family": "serif", "font.size": 11}
FC = dict(teacher="#2CA02C", student="#FF7F0E", neg="#D62728", grey="#7F7F7F")
S_R = 0.62  # horizontal position of level r; t is at 0 and level 0 at 1


TEACHER_SLOPE, STUDENT_SLOPE = 1.5, 0.95


def teacher_curve(s):
    return TEACHER_SLOPE * s - 1.25 * s ** 2


def student_curve(s):
    return STUDENT_SLOPE * s - 0.85 * s ** 2


def level_axis(ax):
    ax.annotate("", xy=(1.08, -0.16), xytext=(-0.06, -0.16),
                arrowprops=dict(arrowstyle="-|>", color="0.35", lw=1.2))
    for s, lab in ((0, "$t$"), (S_R, "$r$"), (1, "$0$")):
        ax.plot([s, s], [-0.16, 1.05], c="0.88", lw=0.8, zorder=0)
        ax.plot([s, s], [-0.18, -0.14], c="0.35", lw=1.2)
        ax.text(s, -0.25, lab, ha="center", va="top", fontsize=13)
    ax.text(1.08, -0.12, "level", ha="right", va="bottom", fontsize=9, color="0.45")
    ax.set_xlim(-0.12, 1.15)
    ax.set_ylim(-0.5, 1.12)
    ax.axis("off")


class GradientLegendLine(Line2D):
    """Legend-only line, drawn in the legend as the c0 -> c1 gradient of gradient_curve."""

    def __init__(self, label, c0="#E8A04F", c1="#4C72B0", lw=4, alpha=None):
        super().__init__([], [], lw=lw, alpha=alpha, label=label)
        self.c0, self.c1 = c0, c1


class _GradientLegendHandler(HandlerBase):
    def create_artists(self, legend, orig, xdescent, ydescent, width, height, fontsize, trans):
        n = 24
        x = np.linspace(-xdescent, width - xdescent, n + 1)
        y = np.full(n + 1, height / 2 - ydescent)
        segs = np.stack([np.stack([x[:-1], y[:-1]], 1), np.stack([x[1:], y[1:]], 1)], 1)
        a = np.linspace(0, 1, n)[:, None]
        colors = (1 - a) * np.array(matplotlib.colors.to_rgb(orig.c0)) + a * np.array(matplotlib.colors.to_rgb(orig.c1))
        return [LineCollection(segs, colors=colors, linewidths=orig.get_linewidth(), alpha=orig.get_alpha(),
                               capstyle="butt", transform=trans)]


Legend.update_default_handler_map({GradientLegendLine: _GradientLegendHandler()})


def gradient_curve(ax, f, s_end, c0="#E8A04F", c1="#4C72B0", lw=4, alpha=1.0, s_stop=None):
    """Curve from 0 to s_end coloured from c0 to c1; with s_stop, only the part up to s_stop is drawn, in the
    colours it has in the full curve."""
    s = np.linspace(0, s_end, 120)
    y = f(s)
    for i in range(len(s) - 1):
        if s_stop is not None and s[i] >= s_stop:
            break
        a = i / (len(s) - 2)
        c = tuple((1 - a) * np.array(matplotlib.colors.to_rgb(c0)) + a * np.array(matplotlib.colors.to_rgb(c1)))
        ax.plot(s[i:i + 2], y[i:i + 2], c=c, lw=lw, alpha=alpha, solid_capstyle="round", zorder=2)


def arrow(ax, p, q, c, ls="-", lw=2.0, z=4):
    ax.annotate("", xy=q, xytext=p, zorder=z,
                arrowprops=dict(arrowstyle="-|>", color=c, lw=lw, ls=ls, shrinkA=0, shrinkB=4, mutation_scale=14))


def dot(ax, p, c, s=55, z=5, **kw):
    ax.scatter([p[0]], [p[1]], c=c, s=s, zorder=z, edgecolors="white", linewidths=0.8, **kw)


@matplotlib.rc_context(FIG_RC)
def f2_tangent_vs_chord():
    fig, axs = plt.subplots(1, 2, figsize=(11.5, 4.9))
    x_t = (0.0, 0.0)
    T_tan = (S_R, TEACHER_SLOPE * S_R)           # teacher tangent point X[x_t, v_teacher, r]
    T_chord = (S_R, teacher_curve(S_R))          # teacher's exact flow map x_r (on its trajectory)
    S_tan = (S_R, STUDENT_SLOPE * S_R)           # student's own tangent point
    S_chord = (S_R, student_curve(S_R))          # student's prediction X[x_t, u_theta, r]

    ax = axs[0]
    level_axis(ax)
    gradient_curve(ax, teacher_curve, 1.0, alpha=0.35)
    gradient_curve(ax, teacher_curve, S_R)
    arrow(ax, x_t, T_chord, "k", lw=2.2)
    arrow(ax, x_t, T_tan, FC["grey"], ls="--", lw=2.0)
    dot(ax, x_t, "k")
    dot(ax, T_chord, "k")
    dot(ax, T_tan, FC["grey"])
    ax.annotate("", xy=(S_R + 0.035, T_tan[1]), xytext=(S_R + 0.035, T_chord[1]),
                arrowprops=dict(arrowstyle="<->", color=FC["neg"], lw=1.3))
    ax.text(S_R + 0.06, (T_tan[1] + T_chord[1]) / 2, "curvature\ngap", color=FC["neg"], fontsize=10, va="center")
    ax.text(T_tan[0] + 0.03, T_tan[1] + 0.02, "tangent point\n$X[x_t, v_{\\mathrm{teacher}}, r]$", ha="left",
            va="bottom", fontsize=11, color="0.25")
    ax.text(T_chord[0] + 0.03, T_chord[1] - 0.13, "flow map $x_r$\n$X[x_t, u, r]$", ha="left", fontsize=11)
    ax.text(x_t[0] + 0.01, x_t[1] - 0.09, "$x_t$", fontsize=13, va="top")
    ax.add_line(GradientLegendLine("teacher ODE trajectory"))
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.02), frameon=False, fontsize=9.5)
    ax.set_title("(a) one teacher call gives a tangent point, not the flow map", fontsize=12, pad=48)

    ax = axs[1]
    level_axis(ax)
    gradient_curve(ax, teacher_curve, S_R, alpha=0.35)
    gradient_curve(ax, student_curve, S_R, c0="#F6C08A", c1=FC["student"], lw=3, alpha=0.55)
    arrow(ax, x_t, T_tan, FC["teacher"], ls="--")
    arrow(ax, x_t, S_tan, FC["neg"], ls="--")
    arrow(ax, x_t, S_chord, FC["student"], lw=2.2)
    dot(ax, x_t, "k")
    dot(ax, T_tan, FC["teacher"])
    dot(ax, S_tan, FC["neg"])
    dot(ax, S_chord, FC["student"])
    dot(ax, T_chord, "white", s=40, z=4)
    ax.scatter([T_chord[0]], [T_chord[1]], facecolors="none", edgecolors="0.45", s=40, zorder=5)
    disp = np.array(T_tan) - np.array(S_tan)
    base = np.array(S_chord) + np.array([0.075, 0.0])   # drawn beside the points it would overlap
    ax.plot([S_chord[0], base[0]], [S_chord[1], base[1]], c="k", lw=0.8, ls=":")
    arrow(ax, tuple(base), tuple(base + disp), "k", lw=2.4, z=6)
    ax.text(base[0] + 0.02, base[1] + disp[1] / 2, "$x_{\\mathrm{pos}} - x_{\\mathrm{neg}}$\napplied to $\\hat x^\\theta_r$",
            fontsize=10.5, va="center")
    ax.text(T_tan[0] - 0.03, T_tan[1] + 0.07, "$x_{\\mathrm{pos}} = X[x_t, v_{\\mathrm{teacher}}, r]$\nteacher's tangent point",
            color=FC["teacher"], fontsize=10.5, ha="right")
    ax.text(S_tan[0] - 0.03, S_tan[1] + 0.07,
            "$x_{\\mathrm{neg}} = X[x_t, v_{\\mathrm{student}}, r]$\nstudent's own tangent point",
            color=FC["neg"], fontsize=10.5, ha="right")
    ax.text(S_chord[0] + 0.02, S_chord[1] - 0.12, "$\\hat x^\\theta_r = X[x_t, u_\\theta, r]$\nstudent's prediction (moved)",
            color=FC["student"], fontsize=10.5, va="center")
    ax.scatter([], [], facecolors="none", edgecolors="0.45", s=40, label="teacher's flow map $x_r$")
    ax.add_line(GradientLegendLine("teacher ODE trajectory", alpha=0.5))
    ax.add_line(GradientLegendLine("student's own trajectory", c0="#F6C08A", c1=FC["student"], lw=3, alpha=0.6))
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.02), ncol=3, frameon=False, fontsize=9.5,
              columnspacing=1.2, handlelength=1.6)
    ax.text(x_t[0] + 0.01, x_t[1] - 0.09, "$x_t$", fontsize=13, va="top")
    ax.text(0.5, 1.02,
            "MeanFlow: $v_{\\mathrm{student}} = u_\\theta + J$, intrinsic, induced by the generator\n"
            "DMD: $v_{\\mathrm{student}} = v_\\phi$, extrinsic, learned by an auxiliary network",
            transform=ax.transAxes, ha="center", va="bottom", fontsize=10.5, linespacing=1.5,
            bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="0.75"))
    ax.set_title("(b) so compare tangent with tangent, and move the student's prediction", fontsize=12, pad=48)
    fig.tight_layout()
    MATERIALS.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "svg"):
        fig.savefig(MATERIALS / f"f2_tangent_vs_chord.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[figures] {MATERIALS / 'f2_tangent_vs_chord.png'}")




@matplotlib.rc_context(FIG_RC)
def f_pull_push_anchors(step=200, k=3):
    """Attraction toward x_pos, repulsion from x_neg and their sum on a few anchors, MeanFlow vs DMD."""
    d = np.load(ART / "assets" / "distribution_demo.npz")
    panels = [
        ("mf-on-all", "MeanFlow",
         "$x_{\\mathrm{pred}} = X[x_t, u_\\theta, r]$: the student's prediction from $x_t$\n"
         "$x_{\\mathrm{pos}}, x_{\\mathrm{neg}}$: tangent points from the same $x_t$"),
        ("dmd-on-all", "DMD",
         "$x_{\\mathrm{pred}}$: the student's own point at level $r$ (its sample at $r=0$)\n"
         "$x_{\\mathrm{pos}}, x_{\\mathrm{neg}}$: teacher's and fake's estimates at renoised $x_{t'}$"),
    ]
    fig, axs = plt.subplots(1, 2, figsize=(12, 6.4))
    for ax, (run, name, explain) in zip(axs, panels):
        p = int(np.searchsorted(d[f"{run}__snap_steps"], step))
        xp, pos, neg = (d[f"{run}__c__{f}"][p] for f in ("x_pred", "x_pos", "x_neg"))
        upd = pos - neg
        size = np.linalg.norm(upd, axis=1)
        chosen = farthest_points(xp, k, size >= np.quantile(size, 0.3))
        att, rep = pos - xp, xp - neg
        scale = max(1.0, round(0.5 / np.median(size[chosen])))
        data = d["data"]
        ax.scatter(data[:, 0], data[:, 1], s=1.2, c="0.82", lw=0, zorder=0)
        for i in chosen:
            ax.plot(*np.stack([xp[i], pos[i]]).T, c=FC["teacher"], lw=0.7, ls=":", zorder=1)
            ax.plot(*np.stack([xp[i], neg[i]]).T, c=FC["neg"], lw=0.7, ls=":", zorder=1)
            arrow(ax, xp[i], xp[i] + att[i], FC["teacher"], lw=2.0)
            arrow(ax, xp[i], xp[i] + rep[i], FC["neg"], lw=2.0)
            arrow(ax, xp[i], xp[i] + scale * upd[i], "k", lw=2.6, z=6)
            dot(ax, pos[i], FC["teacher"], s=60)
            dot(ax, neg[i], FC["neg"], s=60)
            dot(ax, xp[i], "k", s=70, z=7)
        pts = np.concatenate([xp[chosen], pos[chosen], neg[chosen], (xp + rep)[chosen], (xp + scale * upd)[chosen]])
        c, half = (pts.min(0) + pts.max(0)) / 2, max(1.3, (pts.max(0) - pts.min(0)).max() / 2 + 0.45)
        ax.set_xlim(c[0] - half * 1.15, c[0] + half * 1.15)
        ax.set_ylim(c[1] - half, c[1] + half * 1.35)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color("0.8")
        ax.set_title(f"{name}  (step {int(d[f'{run}__snap_steps'][p])}, $r = 0.3$)", fontsize=13)
        ax.text(0.02, 0.02, explain, transform=ax.transAxes, fontsize=9.5, va="bottom",
                bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="0.8"))
        lab = "update $x_{\\mathrm{pos}} - x_{\\mathrm{neg}}$" + (f"  (drawn $\\times${scale:g})" if scale > 1 else "")
        handles = [
            plt.Line2D([], [], marker="o", ls="", c="k", label="$x_{\\mathrm{pred}}$ (moved)"),
            plt.Line2D([], [], marker="o", ls="", c=FC["teacher"], label="$x_{\\mathrm{pos}}$"),
            plt.Line2D([], [], marker="o", ls="", c=FC["neg"], label="$x_{\\mathrm{neg}}$"),
            plt.Line2D([], [], c=FC["teacher"], lw=2, label="attraction  $x_{\\mathrm{pos}} - x_{\\mathrm{pred}}$"),
            plt.Line2D([], [], c=FC["neg"], lw=2, label="repulsion  $x_{\\mathrm{pred}} - x_{\\mathrm{neg}}$"),
            plt.Line2D([], [], c="k", lw=2.6, label=lab),
        ]
        ax.legend(handles=handles, loc="upper left", fontsize=9, frameon=True, framealpha=0.9, edgecolor="0.85")
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(MATERIALS / f"pull_push_anchors.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[figures] {MATERIALS / 'pull_push_anchors.png'}")


def f_teaser(run="dmd-on-all", step=200, k=3):
    """Thumbnail of the post, without text: the student's one-step samples at an early step and, on a few of them,
    the pull toward x_pos (green), the push away from x_neg (red) and the update x_pos - x_neg (black)."""
    d = np.load(ART / "assets" / "distribution_demo.npz")
    p = int(np.searchsorted(d[f"{run}__snap_steps"], step))
    xp, pos, neg = (d[f"{run}__c__{f}"][p] for f in ("x_pred", "x_pos", "x_neg"))
    upd = pos - neg
    size = np.linalg.norm(upd, axis=1)
    chosen = farthest_points(xp, k, size >= np.quantile(size, 0.3))
    scale = max(1.0, round(0.5 / np.median(size[chosen])))
    data, cloud = d["data"], d[f"{run}__k1"][p]
    shown = np.concatenate([xp[chosen], pos[chosen], neg[chosen], (2 * xp - neg)[chosen],
                            (xp + scale * upd)[chosen]])
    lo = np.minimum(np.quantile(data, 0.005, axis=0), shown.min(0)) - 0.12
    hi = np.maximum(np.quantile(data, 0.995, axis=0), shown.max(0)) + 0.12
    width = 5.0
    fig = plt.figure(figsize=(width, width * (hi[1] - lo[1]) / (hi[0] - lo[0])))
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    ax.scatter(*data.T, s=3.0, c="#4C72B0", alpha=0.3, lw=0, zorder=0)
    ax.scatter(*cloud[:1500].T, s=3.0, c=FC["student"], alpha=0.35, lw=0, zorder=1)
    for i in chosen:
        arrow(ax, xp[i], pos[i], FC["teacher"], lw=2.4)
        arrow(ax, xp[i], 2 * xp[i] - neg[i], FC["neg"], lw=2.4)
        arrow(ax, xp[i], xp[i] + scale * upd[i], "k", lw=3.0, z=6)
        dot(ax, pos[i], FC["teacher"], s=60)
        dot(ax, neg[i], FC["neg"], s=60)
        dot(ax, xp[i], "k", s=75, z=7)
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    ax.axis("off")
    for ext in ("png", "svg"):
        fig.savefig(MATERIALS / f"teaser.{ext}", dpi=200, facecolor="white")
    plt.close(fig)
    print(f"[figures] {MATERIALS / 'teaser.png'}")


# ====================================================================================================
# orchestration: what the notebook calls
# ====================================================================================================
def run_teacher(preset_name="smoke", device="cpu", skip_train=False):
    """Train the flow-matching teacher, cache its reference samples and ODE endpoints, and evaluate it."""
    preset, dev = get_preset(preset_name), pick_device(device)
    seed_all(0)
    teacher = load_teacher(dev) if skip_train else train_teacher(preset, dev)
    build_caches(teacher, preset, dev)
    return evaluate_teacher(teacher, preset, dev)


def run_dir(argv):
    a = get_args(argv)
    return (Path(a.root) if a.root else ART / "runs") / run_name(a.variant, a.tag) / f"seed{a.seed}"


def run_jobs(jobs, parallel=12, skip_done=True):
    """Run `python assess.py train <argv>` for every argv in jobs, `parallel` processes at a time.
    A run whose config.json exists is complete and is skipped unless skip_done is False."""
    env = dict(os.environ, OMP_NUM_THREADS="1")
    todo = [j for j in jobs if not (skip_done and (run_dir(j) / "config.json").exists())]
    print(f"[jobs] {len(jobs) - len(todo)} already done, {len(todo)} to run", flush=True)

    def one(argv):
        out = run_dir(argv)
        out.mkdir(parents=True, exist_ok=True)
        with open(out / "stdout.log", "w") as f:
            r = subprocess.run([sys.executable, str(ROOT / "assess.py"), "train", *argv], cwd=ROOT, env=env,
                               stdout=f, stderr=subprocess.STDOUT)
        lines = [l for l in (out / "stdout.log").read_text().splitlines() if l.startswith("[")]
        print(" | ".join(lines[-2:]), flush=True)
        if r.returncode:
            raise RuntimeError(f"{' '.join(argv)} failed, see {out / 'stdout.log'}")

    with ThreadPoolExecutor(parallel) as ex:
        list(ex.map(one, todo))


def table_distribution(root):
    """The table of Section 6: SW2 and KL from the test evaluation of every seed (final_kl.csv), the coupling
    error from each seed's last quick evaluation; means over seeds. Also the four-step KL."""
    rows = read_csv(Path(root).parent / "final_kl.csv")
    mean = lambda run, K, k: np.mean([r[k] for r in rows if r["run"] == run and r["K"] == K])
    coupling = lambda run: np.mean([read_csv(q)[-1]["coupling_k1"]
                                    for q in sorted(Path(root, run).glob("seed*/quick_log.csv"))])
    print("| Setup | SW2, 1 step | Coupling, 1 step | KL(student||data), 1 step | KL, 2 steps | KL, 4 steps |")
    print("|---|---|---|---|---|---|")
    for run, label in DEMO:
        print(f"| {label} | {mean(run, 1, 'sw2'):.3f} | {coupling(run):.3f} | {mean(run, 1, 'kl_rev'):.2f} | "
              f"{mean(run, 2, 'kl_rev'):.2f} | {mean(run, 4, 'kl_rev'):.2f} |")


def table_kernel():
    """The table of Section 8: one-step KL(student || data) / one-step SW2 for every N and family."""
    ds = {f: np.load(ART / "assets" / f"{f}_demo.npz") for f in FAMILIES}
    print("| N | " + " | ".join(FAMILIES[f]["title"] for f in ds) + " |")
    print("|---" * (len(ds) + 1) + "|")
    for N in NS:
        cells = [f"{d[f'{N}__final_kl_rev_k1'].mean():.2f} / {d[f'{N}__final_sw2_k1'].mean():.3f}" for d in ds.values()]
        print(f"| {N} | " + " | ".join(cells) + " |")


def run_distribution(preset_name="smoke", device="cpu", steps=3000, batch=None, snap_n=2000, seeds=(0, 1, 2),
                     parallel=12, fps=8, hold=12):
    """The six students of Section 6 on the dataset selected by CG_DATA: the teacher (reused if trained), the
    runs, the export, the table and the sample animations, written to materials/<data>_samples_k{1,2}."""
    run_teacher(preset_name, device, skip_train=(TDIR / "teacher.pt").exists())
    root = ART / "distribution" / "runs"
    common = ["--preset", preset_name, "--device", device, "--steps", str(steps), "--snap-count", "61",
              "--snap-n", str(snap_n), *(["--batch", str(batch)] if batch else [])]
    run_jobs([["--variant", run, "--seed", str(s), "--root", str(root), *common] for run, _ in DEMO for s in seeds],
             parallel)
    d = np.load(export_distribution(root, preset_name, pick_device(device)))
    meta = json.loads(str(d["meta"]))
    table_distribution(root)
    prefix = "" if DATA == "moons" else f"{DATA}_"
    for K in (1, 2):
        anim_samples(d, meta, K, fps, hold, out_name=f"{prefix}samples_k{K}")


def sw2_curves(root, runs, steps=(0, 200, 500, 1000, 2000, 3000)):
    """One-step SW2 to the teacher's samples at a few training steps, mean over seeds (quick evaluations)."""
    print(f"{'run':28s}" + "".join(f"{s:>10d}" for s in steps))
    for run in runs:
        qs = [read_csv(q) for q in sorted(Path(root, run).glob("seed*/quick_log.csv"))]
        vals = [np.mean([next((r["sw2_k1"] for r in q if r["step"] == s), np.nan) for q in qs]) for s in steps]
        print(f"{run:28s}" + "".join(f"{v:10.3f}" for v in vals))


def cli(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ["teacher"]:
        p = argparse.ArgumentParser(prog="assess.py teacher")
        p.add_argument("--preset", default="smoke")
        p.add_argument("--device", default="cpu")
        p.add_argument("--skip-train", action="store_true")
        a = p.parse_args(argv[1:])
        run_teacher(a.preset, a.device, a.skip_train)
    elif argv[:1] == ["train"]:
        train_student(get_args(argv[1:]))
    elif argv[:1] == ["distribution"]:
        p = argparse.ArgumentParser(prog="assess.py distribution")
        p.add_argument("--preset", default="smoke")
        p.add_argument("--device", default="cpu")
        p.add_argument("--steps", type=int, default=3000)
        p.add_argument("--batch", type=int, default=None, help="student batch (default: the preset's)")
        p.add_argument("--snap-n", type=int, default=2000, help="samples per animation frame")
        p.add_argument("--parallel", type=int, default=12)
        a = p.parse_args(argv[1:])
        run_distribution(a.preset, a.device, a.steps, a.batch, a.snap_n, parallel=a.parallel)
    else:
        raise SystemExit("usage: python assess.py teacher [...] | python assess.py train --variant ... [...] | "
                         "CG_DATA=checker3d python assess.py distribution [...]")


if __name__ == "__main__":
    cli()
