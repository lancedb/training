"""The scoring maths exactly as detect.py computes it: one pass over the whole frames table in
numpy, per-episode slices by contiguous row ranges. Used as the reference for the equivalence test."""
import numpy as np

MAX_LAG = 10


def fsl(col):
    col = col.combine_chunks()
    return col.flatten().to_numpy(zero_copy_only=False).reshape(len(col), -1).astype(np.float32)


def robust_z(x):
    med = np.median(x)
    mad = np.median(np.abs(x - med)) * 1.4826 + 1e-9
    return (x - med) / mad


def score_all(T, n_action_dims=6, n_pos_dims=3):
    """T: pyarrow table with index, episode_index, task_index, action, observation_state, in row order.
    Returns per-frame jerk and a dict of per-episode arrays, exactly as detect.py."""
    ep = T.column("episode_index").to_numpy()
    task = T.column("task_index").to_numpy()
    act = fsl(T.column("action"))
    st = fsl(T.column("observation_state"))
    episodes, first = np.unique(ep, return_index=True)
    order = np.argsort(first)
    episodes = episodes[order]
    ep_from = first[order]
    ep_to = np.append(ep_from[1:], len(ep))

    jerk = np.zeros(len(ep), dtype=np.float32)
    agree = np.zeros(len(episodes), dtype=np.float32)
    lag_best = np.zeros(len(episodes), dtype=np.int64)
    lag_gain = np.zeros(len(episodes), dtype=np.float32)
    ep_task = np.zeros(len(episodes), dtype=np.int64)
    for i, (s, e) in enumerate(zip(ep_from, ep_to)):
        A = act[s:e]
        d = np.abs(np.diff(A[:, :n_action_dims], axis=0)).sum(axis=1)
        jerk[s + 1:e] = d
        jerk[s] = d[0] if len(d) else 0
        if e - s > MAX_LAG + 5:
            dx = st[s + 1:e, :n_pos_dims] - st[s:e - 1, :n_pos_dims]
            cors = []
            for lag in range(MAX_LAG + 1):
                cs = []
                for k in range(n_pos_dims):
                    x, y = A[: len(dx) - lag, k], dx[lag:, k]
                    if x.std() > 1e-6 and y.std() > 1e-6:
                        cs.append(np.corrcoef(x, y)[0, 1])
                cors.append(float(np.mean(cs)) if cs else 0.0)
            agree[i] = cors[0]
            lag_best[i] = int(np.argmax(cors))
            lag_gain[i] = max(cors) - cors[0]
        ep_task[i] = task[s]
    ep_jerk = np.array([jerk[s:e].mean() for s, e in zip(ep_from, ep_to)], dtype=np.float32)
    return jerk, dict(episodes=episodes, task=ep_task, jerk_ep=ep_jerk, agree=agree,
                      lag_best=lag_best, lag_gain=lag_gain)


def goal_dist_all(E, ep_task):
    """E: (n_episodes, dim) unit embeddings. Distance to the median of the OTHER same-task episodes."""
    goal_dist = np.zeros(len(E), dtype=np.float32)
    for t in np.unique(ep_task):
        m = np.where(ep_task == t)[0]
        for i in m:
            others = E[m[m != i]]
            ref = np.median(others, axis=0)
            ref /= (np.linalg.norm(ref) + 1e-9)
            goal_dist[i] = 1.0 - float(E[i] @ ref)
    return goal_dist
