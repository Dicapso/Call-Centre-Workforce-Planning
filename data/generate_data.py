"""Synthetic call-level data for a Baku utility company's customer hotline (seed 42).

10 weeks (27 Jul - 4 Oct 2026), open 08:00-20:00 every day. Arrivals are a non-homogeneous
Poisson process (weekday x intraday shape, billing-cycle uplift, two outage incidents).
Waits and abandons come from an FCFS queue simulation with the actual logged-in agents.
A few data-quality issues are injected on purpose.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from wfm import simulate_day, NI, IV, OPEN_H  # noqa: E402

rng = np.random.default_rng(42)
START, DAYS = pd.Timestamp("2026-07-27"), 70
BASE_WEEKDAY_CALLS = 2650
DOW = [1.22, 1.06, 1.00, 0.98, 0.95, 0.62, 0.45]
SHAPE_WD = np.array([.45, .65, .95, 1.15, 1.40, 1.50, 1.45, 1.35, 1.15, 1.00, 1.05, 1.20,
                     1.30, 1.30, 1.20, 1.10, 1.00, .95, .85, .75, .60, .50, .40, .30])
SHAPE_WE = np.array([.30, .50, .80, 1.0, 1.20, 1.30, 1.35, 1.35, 1.30, 1.25, 1.20, 1.15,
                     1.10, 1.05, 1.00, .95, .90, .85, .80, .70, .60, .50, .40, .30])
REASONS = {  # share, mean handle seconds
    "billing": (.30, 240), "payment": (.15, 150), "outage": (.12, 180), "meter_reading": (.15, 200),
    "new_connection": (.10, 420), "complaint": (.08, 480), "other": (.10, 260)}
INCIDENTS = {"2026-08-18": 10, "2026-09-15": 12}   # outage incident start interval (13:00, 14:00)
# current roster: (start interval, length in intervals, agents)
SHIFTS_WD = [(0, 18, 14), (2, 18, 12), (6, 18, 12)]   # 08-17, 09-18, 11-20
SHIFTS_SAT = [(0, 18, 9), (6, 18, 8)]
SHIFTS_SUN = [(0, 18, 7), (6, 18, 6)]

names = list(REASONS)
shares = np.array([REASONS[r][0] for r in names])
mean_ht = np.array([REASONS[r][1] for r in names])
SIG = 0.6


def roster_for(dow):
    shifts = SHIFTS_WD if dow < 5 else SHIFTS_SAT if dow == 5 else SHIFTS_SUN
    sched = np.zeros(NI, int)
    for s, L, c in shifts:
        sched[s:s + L] += c
    return sched


calls, roster = [], []
for d in range(DAYS):
    date = START + pd.Timedelta(days=d)
    dow = date.dayofweek
    shape = SHAPE_WD if dow < 5 else SHAPE_WE
    bill = 1.15 if date.day <= 5 else 1.05 if date.day <= 10 else 1.0
    level = BASE_WEEKDAY_CALLS * DOW[dow] * bill * (1 + 0.004 * d / 7) * rng.lognormal(0, 0.05)
    rate = level * shape / shape.sum()
    n_int = rng.poisson(rate)
    extra = np.zeros(NI, int)
    if str(date.date()) in INCIDENTS:
        s0 = INCIDENTS[str(date.date())]
        surge = 2.2 * rate[s0:] * np.exp(-0.18 * np.arange(NI - s0))
        extra[s0:] = rng.poisson(surge)
    t, reason = [], []
    for i in range(NI):
        t.append(rng.uniform(i * IV, (i + 1) * IV, n_int[i] + extra[i]))
        r = rng.choice(len(names), n_int[i], p=shares)
        reason.append(np.concatenate([r, np.full(extra[i], names.index("outage"))]))
    t, reason = np.concatenate(t), np.concatenate(reason)
    order = np.argsort(t)
    t, reason = t[order], reason[order]
    m = mean_ht[reason]
    handle = rng.lognormal(np.log(m) - SIG ** 2 / 2, SIG)
    patience = np.where(rng.random(len(t)) < 0.06, rng.uniform(0, 5, len(t)),
                        rng.exponential(160, len(t)))
    sched = roster_for(dow)
    in_shift = np.full(NI, 0.88)
    in_shift[8:14] = 0.76                     # lunch breaks 12:00-15:00
    absent = rng.uniform(0.02, 0.10)
    logged = rng.binomial(sched, in_shift * (1 - absent))
    wait, ab = simulate_day(t, handle, patience, logged)
    ts = date + pd.Timedelta(hours=OPEN_H) + pd.to_timedelta(t, unit="s")
    calls.append(pd.DataFrame({
        "arrival_ts": ts.floor("s"), "reason": np.array(names)[reason],
        "wait_sec": np.round(wait).astype(int), "abandoned": ab.astype(int),
        "handle_sec": np.where(ab, np.nan, np.round(handle))}))
    roster.append(pd.DataFrame({"date": date.date(), "interval_start": [
        f"{OPEN_H + i // 2:02d}:{30 * (i % 2):02d}" for i in range(NI)],
        "scheduled_agents": sched, "logged_in_agents": logged}))

calls = pd.concat(calls, ignore_index=True)
calls.insert(0, "call_id", [f"C{i:07d}" for i in range(1, len(calls) + 1)])

# ---- inject data-quality issues
n = len(calls)
ans = calls.index[calls.abandoned == 0]
calls.loc[rng.choice(ans, 35, replace=False), "handle_sec"] = rng.choice([0, -1, -30, 3], 35)
calls.loc[rng.choice(ans, 12, replace=False), "handle_sec"] = rng.uniform(4, 9, 12).round() * 3600
calls.loc[rng.choice(n, int(0.015 * n), replace=False), "reason"] = np.nan
test_days = START + pd.to_timedelta(rng.integers(0, DAYS, 28), unit="D")
tests = pd.DataFrame({
    "call_id": [f"T{i:06d}" for i in range(28)],
    "arrival_ts": test_days + pd.to_timedelta(rng.choice([6.2, 7.1, 7.6, 20.4, 21.9], 28), unit="h"),
    "reason": "TEST", "wait_sec": 0, "abandoned": 0, "handle_sec": 30.0})
calls = pd.concat([calls, tests, calls.sample(110, random_state=7)], ignore_index=True)
calls = calls.sort_values("arrival_ts").reset_index(drop=True)

calls.to_csv(Path(__file__).parent / "calls.csv", index=False)
pd.concat(roster).to_csv(Path(__file__).parent / "roster.csv", index=False)
print(f"calls: {len(calls):,} rows, abandon rate {calls.abandoned.mean():.1%}")
