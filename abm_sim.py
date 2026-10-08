"""Стилизованная агентная модель: N вендоров, логит-спрос, табличный Q-learning.
Доля M агентов использует ОДНУ общую Q-таблицу (идентичный алгоритм, общие данные),
остальные учатся независимо. Исследуем волатильность средней цены."""
import numpy as np, json, sys

N = 10
PRICES = np.array([1.2, 1.4, 1.6, 1.8, 2.0])
K = len(PRICES)
C = 1.0; A = 2.0; A0 = 0.0; MU = 0.25
ALPHA, GAMMA, EPS = 0.05, 0.95, 0.10
T_TOTAL, T_MEASURE = 60000, 10000
SHOCK = 0.10  # std экзогенного шока спроса (общий для рынка)


def profits(p, shock):
    u = np.exp((A + shock - p) / MU)
    share = u / (u.sum() + np.exp(A0 / MU))
    return (p - C) * share


def run(n_shared, seed):
    rng = np.random.default_rng(seed)
    # индекс Q-таблицы для каждого агента: общий (0) для первых n_shared, свой для остальных
    table_id = np.array([0] * n_shared + list(range(1, N - n_shared + 1)))
    Q = np.zeros((N - n_shared + 1, K, K))  # [таблица, состояние, действие]
    state = 2
    mean_prices = []
    for t in range(T_TOTAL):
        qrows = Q[table_id, state]                       # (N, K)
        greedy = qrows.argmax(axis=1)
        explore = rng.random(N) < EPS
        a = np.where(explore, rng.integers(0, K, N), greedy)
        p = PRICES[a]
        r = profits(p, rng.normal(0, SHOCK))
        ns = int(np.clip(np.rint((p.mean() - PRICES[0]) / 0.2), 0, K - 1))
        for i in range(N):
            tid = table_id[i]
            td = r[i] + GAMMA * Q[tid, ns].max() - Q[tid, state, a[i]]
            Q[tid, state, a[i]] += ALPHA * td
        state = ns
        mean_prices.append(p.mean())
    mp = np.array(mean_prices[-T_MEASURE:])
    return mp.std(), np.abs(np.diff(mp)).mean(), mp.mean()


if __name__ == "__main__":
    SEEDS = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    out = {}
    for n_shared in [0, 2, 4, 6, 8]:
        res = np.array([run(n_shared, s) for s in range(SEEDS)])
        out[n_shared] = {
            "std_mean": res[:, 0].mean(), "std_sd": res[:, 0].std(ddof=1),
            "absdiff_mean": res[:, 1].mean(), "price_mean": res[:, 2].mean(),
            "std_all": res[:, 0].tolist(),
        }
        print(n_shared, out[n_shared]["std_mean"], out[n_shared]["std_sd"], flush=True)
    json.dump(out, open("/tmp/abm_results.json", "w"), indent=1)
