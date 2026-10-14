"""Workforce-management helpers: cleaning, Erlang C, queue simulation and MILP shift scheduling."""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import milp, LinearConstraint, Bounds

DATA = Path(__file__).parent / "data"
OPEN_H, CLOSE_H = 8, 20          # contact centre open 08:00-20:00, every day
IV = 1800                        # 30-minute intervals
NI = (CLOSE_H - OPEN_H) * 3600 // IV
SL_T = 20                        # service-level threshold, seconds (target 80/20)
SHORT_ABANDON = 5                # abandons within 5 s are excluded from SL
LABELS = [f"{OPEN_H + i // 2:02d}:{30 * (i % 2):02d}" for i in range(NI)]


# ---------------------------------------------------------------- data
def load_raw():
    calls = pd.read_csv(DATA / "calls.csv", parse_dates=["arrival_ts"])
    roster = pd.read_csv(DATA / "roster.csv", parse_dates=["date"])
    return calls, roster


def clean(calls, roster):
    log = {"Rows in": len(calls)}
    df = calls.drop_duplicates()
    log["Exact duplicate rows dropped"] = len(calls) - len(df)
    sec = (df.arrival_ts - df.arrival_ts.dt.normalize()).dt.total_seconds()
    out = (sec < OPEN_H * 3600) | (sec >= CLOSE_H * 3600) | df.reason.eq("TEST")
    log["Test / out-of-hours calls dropped"] = int(out.sum())
    df = df[~out].copy()
    bad_ht = df.abandoned.eq(0) & ~df.handle_sec.between(10, 3 * 3600)
    log["Impossible handle times (<10 s or >3 h) set to missing"] = int(bad_ht.sum())
    df.loc[bad_ht, "handle_sec"] = np.nan
    log["Missing reason -> 'unknown'"] = int(df.reason.isna().sum())
    df["reason"] = df.reason.fillna("unknown")
    df["date"] = df.arrival_ts.dt.normalize()
    df["t"] = (df.arrival_ts - df.date).dt.total_seconds() - OPEN_H * 3600
    df["interval"] = (df.t // IV).astype(int)
    df["dow"] = df.date.dt.dayofweek
    df["short_abandon"] = df.abandoned.eq(1) & (df.wait_sec < SHORT_ABANDON)
    df["in_sl"] = df.abandoned.eq(0) & (df.wait_sec <= SL_T)
    df = df.sort_values("arrival_ts").reset_index(drop=True)
    log["Rows out"] = len(df)
    roster = roster.copy()
    roster["interval"] = roster.groupby("date").cumcount()
    return df, roster, log


def interval_stats(df):
    g = df.groupby(["date", "interval"])
    s = pd.DataFrame({
        "offered": g.size(),
        "abandoned": g.abandoned.sum(),
        "short_ab": g.short_abandon.sum(),
        "in_sl": g.in_sl.sum(),
        "handle_sum": g.handle_sec.sum(),
        "answered": g.abandoned.apply(lambda a: (a == 0).sum()),
    })
    s["sl"] = s.in_sl / (s.offered - s.short_ab).clip(lower=1)
    return s


# ---------------------------------------------------------------- Erlang C
def erlang_c(A, N):
    """Probability an arriving call waits (Erlang C) for offered load A erlangs and N agents."""
    if N <= A:
        return 1.0
    b = 1.0
    for k in range(1, int(N) + 1):          # Erlang B recursion, numerically stable
        b = A * b / (k + A * b)
    return N * b / (N - A * (1 - b))


def service_level(calls, aht, N, T=SL_T, iv=IV):
    A = calls * aht / iv
    if N <= A:
        return 0.0
    return 1 - erlang_c(A, N) * np.exp(-(N - A) * T / aht)


def asa(calls, aht, N, iv=IV):
    A = calls * aht / iv
    return np.inf if N <= A else erlang_c(A, N) * aht / (N - A)


def required_agents(calls, aht, target=0.80, T=SL_T, max_occ=0.90, iv=IV):
    """Smallest N meeting the service-level target and an occupancy cap."""
    if calls <= 0:
        return 1
    A = calls * aht / iv
    N = max(1, int(np.ceil(A / max_occ)))
    while service_level(calls, aht, N, T, iv) < target:
        N += 1
    return N


# ---------------------------------------------------------------- simulation
def simulate_day(t, handle, patience, agents):
    """FCFS multi-server queue with abandonment and an agent count that changes every interval.

    t: arrival seconds after opening (sorted); agents: logged-in agents per interval.
    Returns wait seconds (time to abandon for abandoned calls) and an abandoned flag.
    """
    agents = np.asarray(agents, int)
    m = max(int(agents.max()), 1)
    # earliest time >= start of interval i at which agent k is logged in
    na = np.full((m, NI + 1), np.inf)
    for k in range(m):
        nxt = np.inf
        for i in range(NI - 1, -1, -1):
            if agents[i] > k:
                nxt = i * IV
            na[k, i] = nxt
    free = np.zeros(m)
    ks = np.arange(m)
    wait = np.empty(len(t))
    ab = np.zeros(len(t), bool)
    for j in range(len(t)):
        s = np.maximum(free, t[j])
        idx = np.minimum((s // IV).astype(int), NI)
        start = np.maximum(s, na[ks, idx])
        k = int(np.argmin(start))
        w = start[k] - t[j]
        if w > patience[j]:
            ab[j], wait[j] = True, patience[j]
        else:
            wait[j] = w
            free[k] = start[k] + handle[j]
    return wait, ab


def sl_from_sim(wait, ab, T=SL_T):
    short = ab & (wait < SHORT_ABANDON)
    return ((~ab) & (wait <= T)).sum() / max((~short).sum(), 1)


# ---------------------------------------------------------------- scheduling
def schedule_milp(req, ft_len=16, pt_len=8, pt_max_share=0.30, min_ft=0):
    """Minimise paid agent-hours so that scheduled agents >= req in every interval.

    Full-time shifts: ft_len intervals; part-time: pt_len intervals; any start that ends by closing.
    Part-time hours are capped at pt_max_share of all hours.
    """
    req = np.asarray(req, float)
    shifts = [("FT", s, ft_len) for s in range(NI - ft_len + 1)] + \
             [("PT", s, pt_len) for s in range(NI - pt_len + 1)]
    n = len(shifts)
    cover = np.zeros((NI, n))
    for j, (_, s, L) in enumerate(shifts):
        cover[s:s + L, j] = 1
    hours = np.array([L / 2 for _, _, L in shifts])
    is_pt = np.array([k == "PT" for k, _, _ in shifts])
    # PT hours - share * total hours <= 0
    pt_row = hours * (is_pt - pt_max_share)
    ft_row = (~is_pt).astype(float)
    cons = [LinearConstraint(cover, req, np.inf),
            LinearConstraint(pt_row[None, :], -np.inf, 0),
            LinearConstraint(ft_row[None, :], min_ft, np.inf)]
    res = milp(c=hours, constraints=cons, integrality=np.ones(n), bounds=Bounds(0, np.inf))
    if not res.success:
        raise RuntimeError(res.message)
    x = np.round(res.x).astype(int)
    plan = pd.DataFrame([(k, LABELS[s], L / 2, c) for (k, s, L), c in zip(shifts, x) if c > 0],
                        columns=["type", "start", "hours", "agents"])
    return plan, (cover @ x).round().astype(int), float(hours @ x)

def schedule_budget(req_by_day, budget_hours, ft_len=16, pt_len=8, pt_max_share=0.30):
    """Fixed paid-hours budget for the whole week: minimise total agent-interval shortfall (weighted by
    the requirement, so peaks matter) while covering as much of the requirement as the budget allows.

    req_by_day: list of requirement arrays (one per day). Returns per-day coverage arrays and hours used.
    """
    D = len(req_by_day)
    shifts = [("FT", s, ft_len) for s in range(NI - ft_len + 1)] + [("PT", s, pt_len) for s in range(NI - pt_len + 1)]
    ns = len(shifts)
    cover = np.zeros((NI, ns))
    for j, (_, s, L) in enumerate(shifts):
        cover[s:s + L, j] = 1
    hours = np.array([L / 2 for _, _, L in shifts])
    is_pt = np.array([k == "PT" for k, _, _ in shifts])
    nx, nu = D * ns, D * NI
    # variables: x[d, j] (integer), u[d, i] (continuous shortfall)
    A_cov = np.zeros((nu, nx + nu))
    lb = np.zeros(nu)
    for d in range(D):
        A_cov[d * NI:(d + 1) * NI, d * ns:(d + 1) * ns] = cover
        A_cov[d * NI:(d + 1) * NI, nx + d * NI: nx + (d + 1) * NI] = np.eye(NI)
        lb[d * NI:(d + 1) * NI] = req_by_day[d]
    row_h = np.concatenate([np.tile(hours, D), np.zeros(nu)])
    row_pt = np.concatenate([np.tile(hours * (is_pt - pt_max_share), D), np.zeros(nu)])
    cons = [LinearConstraint(A_cov, lb, np.inf),
            LinearConstraint(row_h[None, :], -np.inf, budget_hours),
            LinearConstraint(row_pt[None, :], -np.inf, 0)]
    # shortfall below requirement in an interval is penalised more when the requirement is large
    w = np.concatenate([np.asarray(r, float) ** 2 / np.maximum(np.asarray(r, float), 1) for r in req_by_day])
    c = np.concatenate([np.zeros(nx), w])
    integ = np.concatenate([np.ones(nx), np.zeros(nu)])
    res = milp(c=c, constraints=cons, integrality=integ, bounds=Bounds(0, np.inf),
               options={"time_limit": 60})
    x = np.round(res.x[:nx]).astype(int).reshape(D, ns)
    cov = (x @ cover.T).round().astype(int)
    return cov, float((x * hours).sum())
