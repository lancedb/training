"""The same three detectors written as per-episode features on an `episodes` table.

Each UDF takes one episode_index, reads that episode's frames from the frames table (the same
pattern EmbedFromBlob uses with the videos table), and returns one value. Nothing is merged and
nothing is broadcast onto frames.
"""
import numpy as np
import pyarrow as pa
import lancedb
from geneva import udf

MAX_LAG = 10


def _fsl(col):
    col = col.combine_chunks()
    return col.flatten().to_numpy(zero_copy_only=False).reshape(len(col), -1).astype(np.float32)


class _EpisodeReader:
    """Opens the frames table once per worker and hands back one episode's actions and states."""

    def __init__(self, root, storage_options=None):
        self.root, self.storage_options = root, storage_options
        self._tbl = None

    def episode(self, episode_index):
        if self._tbl is None:
            self._tbl = lancedb.connect(self.root, storage_options=self.storage_options).open_table("frames")
        t = (self._tbl.search().where(f"episode_index = {int(episode_index)}", prefilter=True)
             .select(["frame_index", "action", "observation_state"]).limit(1_000_000).to_arrow())
        t = t.sort_by("frame_index")
        return _fsl(t.column("action")), _fsl(t.column("observation_state"))


def make_udfs(root, storage_options=None, n_action_dims=6, n_pos_dims=3):
    """Build the UDFs bound to one dataset root. Returned as plain callables carrying geneva metadata."""

    @udf(data_type=pa.float32(), input_columns=["episode_index"])
    class EpisodeJerk:
        """Mean over the episode of |action_t - action_{t-1}| summed over the arm dims."""

        def __init__(self):
            self.reader = _EpisodeReader(root, storage_options)

        def __call__(self, episode_index: int) -> float:
            A, _ = self.reader.episode(episode_index)
            d = np.abs(np.diff(A[:, :n_action_dims], axis=0)).sum(axis=1)
            if len(d) == 0:
                return 0.0
            # detect.py assigns d[0] to the first frame as well, then takes the mean over all frames
            return float((d.sum() + d[0]) / len(A))

    @udf(data_type=pa.struct([("agree", pa.float32()), ("lag", pa.int64()), ("gain", pa.float32())]),
         input_columns=["episode_index"])
    class ActLag:
        """At which lag does the commanded translation best explain the observed displacement?"""

        def __init__(self):
            self.reader = _EpisodeReader(root, storage_options)

        def __call__(self, episode_index: int) -> dict:
            A, S = self.reader.episode(episode_index)
            n = len(A)
            if n <= MAX_LAG + 5:
                return {"agree": 0.0, "lag": 0, "gain": 0.0}
            dx = S[1:, :n_pos_dims] - S[:-1, :n_pos_dims]
            cors = []
            for lag in range(MAX_LAG + 1):
                cs = []
                for k in range(n_pos_dims):
                    x, y = A[: len(dx) - lag, k], dx[lag:, k]
                    if x.std() > 1e-6 and y.std() > 1e-6:
                        cs.append(np.corrcoef(x, y)[0, 1])
                cors.append(float(np.mean(cs)) if cs else 0.0)
            return {"agree": float(cors[0]), "lag": int(np.argmax(cors)), "gain": float(max(cors) - cors[0])}

    return EpisodeJerk, ActLag


def make_goal_dist_udf(episodes_uri, emb_col="goal_emb"):
    """Second stage over the episodes table: distance of an episode's final-frame embedding to the
    median of every OTHER same-task episode. The per-task reference set is loaded once per worker."""

    @udf(data_type=pa.float32(), input_columns=["episode_index", "task_index", emb_col])
    class GoalDist:
        def __init__(self):
            self._by_task = None

        def _load(self):
            import lance
            t = lance.dataset(episodes_uri).to_table(columns=["episode_index", "task_index", emb_col])
            E = _fsl(t.column(emb_col))
            ep = t.column("episode_index").to_numpy()
            task = t.column("task_index").to_numpy()
            self._by_task = {int(k): (ep[task == k], E[task == k]) for k in np.unique(task)}

        def __call__(self, episode_index: int, task_index: int, goal_emb) -> float:
            if self._by_task is None:
                self._load()
            eps, E = self._by_task[int(task_index)]
            others = E[eps != int(episode_index)]
            ref = np.median(others, axis=0)
            ref /= (np.linalg.norm(ref) + 1e-9)
            e = np.asarray(goal_emb, dtype=np.float32)
            return float(1.0 - e @ ref)

    return GoalDist
