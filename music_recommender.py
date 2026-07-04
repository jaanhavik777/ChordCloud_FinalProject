"""
Standalone music recommendation module.

Implements the pieces referenced in the project description that were
missing from the original Streamlit app:

- User-based and item-based Collaborative Filtering, trained on a public
  Last.fm user-artist listening-history dataset.
- Content-based filtering using Spotify Web API audio features
  (danceability, energy, tempo, valence, acousticness, ...).
- A simple hybrid recommender blending both signals.
- Offline evaluation (precision@k / recall@k / hit-rate@k via
  leave-one-out cross-validation) so recommendation quality claims are
  backed by a real, reproducible number instead of an invented stat.

This module is intentionally decoupled from the Streamlit app (app.py),
which uses a separate LlamaIndex + Spotify-search pipeline. It's meant to
be run and evaluated on its own, e.g. from the accompanying notebook.
"""

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------------------------------------------------------
# Collaborative Filtering
# ---------------------------------------------------------------------------


class CollaborativeFilteringRecommender:
    """
    User-based and item-based collaborative filtering over a binary
    user-artist listening matrix (rows = users, columns = artists,
    values = 1 if the user listened to the artist).
    """

    def __init__(self, listening_matrix: pd.DataFrame):
        """
        :param listening_matrix: DataFrame indexed by user id, one column
            per artist, binary values.
        """
        self.matrix = listening_matrix
        self._user_similarity: Optional[pd.DataFrame] = None

    @property
    def user_similarity(self) -> pd.DataFrame:
        """Lazily-computed user-user cosine similarity matrix."""
        if self._user_similarity is None:
            sim = cosine_similarity(self.matrix.values)
            self._user_similarity = pd.DataFrame(
                sim, index=self.matrix.index, columns=self.matrix.index
            )
        return self._user_similarity

    def recommend_for_user(
        self, user_id, n_neighbors: int = 10, top_k: int = 10
    ) -> List[str]:
        """
        Recommend artists for a user based on what similar users listen to,
        excluding artists the user already listens to.
        """
        if user_id not in self.matrix.index:
            raise KeyError(f"Unknown user_id: {user_id}")

        sims = self.user_similarity.loc[user_id].drop(user_id).sort_values(
            ascending=False
        )
        neighbor_weights = sims.head(n_neighbors)
        neighbor_ids = neighbor_weights.index

        # Weighted sum of neighbors' listening vectors = affinity score per artist
        weighted_scores = self.matrix.loc[neighbor_ids].T.dot(neighbor_weights)

        already_listened = self.matrix.loc[user_id]
        weighted_scores = weighted_scores[already_listened == 0]

        return weighted_scores.sort_values(ascending=False).head(top_k).index.tolist()

    def recommend_similar_artists(self, artist: str, top_k: int = 10) -> List[str]:
        """Item-based: artists most similar to a given artist by co-listenership."""
        if artist not in self.matrix.columns:
            raise KeyError(f"Unknown artist: {artist}")

        artist_matrix = self.matrix.T  # artists x users
        sim = cosine_similarity(artist_matrix.values)
        sim_df = pd.DataFrame(
            sim, index=artist_matrix.index, columns=artist_matrix.index
        )
        return (
            sim_df[artist].drop(artist).sort_values(ascending=False).head(top_k).index.tolist()
        )


# ---------------------------------------------------------------------------
# Content-Based Filtering (Spotify audio features)
# ---------------------------------------------------------------------------

AUDIO_FEATURE_COLUMNS = [
    "danceability",
    "energy",
    "tempo",
    "valence",
    "acousticness",
    "instrumentalness",
    "speechiness",
    "liveness",
]


class ContentBasedRecommender:
    """
    Content-based filtering using Spotify audio-feature vectors
    (danceability, energy, tempo, valence, acousticness, ...).

    IMPORTANT — data source: Spotify deprecated the `audio_features` /
    `audio_analysis` Web API endpoints for new third-party applications on
    November 27, 2024. Apps without a pre-existing extended-mode quota now
    get a 403 on that endpoint, and Spotify has not announced a
    replacement. Because of that, this class is built to run on a public,
    precomputed dataset of Spotify audio features (~232k tracks, sourced
    from a Spotify-catalog export predating the deprecation) rather than
    a live API call. Use `ContentBasedRecommender.from_csv(...)`.

    If you happen to have an app with grandfathered access to the live
    endpoint, `get_live_audio_features` is kept below so you can still
    pull fresh data for tracks not present in the dataset — just be aware
    it will 403 for most developers today.
    """

    def __init__(self, features_df: pd.DataFrame, metadata: Optional[pd.DataFrame] = None):
        """
        :param features_df: DataFrame indexed by track_id, columns =
            AUDIO_FEATURE_COLUMNS, already normalized to comparable scales.
        :param metadata: optional DataFrame indexed by track_id with
            columns like track_name/artist_name/genre, for lookups and
            human-readable output.
        """
        self.features = features_df
        self.metadata = metadata

    @classmethod
    def from_csv(cls, path: str) -> "ContentBasedRecommender":
        """
        Build the recommender from a precomputed Spotify audio-features
        CSV (e.g. the ~232k-track public dataset used in this project).
        Expects at minimum: track_id, danceability, energy, tempo,
        valence, acousticness, instrumentalness, speechiness, liveness.
        Optional metadata columns (track_name, artist_name, genre) are
        kept separately for lookups/display.
        """
        raw = pd.read_csv(path)
        raw = raw.drop_duplicates(subset="track_id").set_index("track_id")

        features = raw[AUDIO_FEATURE_COLUMNS].copy()
        tempo_range = features["tempo"].max() - features["tempo"].min()
        features["tempo"] = (features["tempo"] - features["tempo"].min()) / (
            tempo_range + 1e-9
        )

        metadata_cols = [c for c in ("track_name", "artist_name", "genre") if c in raw.columns]
        metadata = raw[metadata_cols].copy() if metadata_cols else None

        return cls(features, metadata)

    def find_track_id(self, track_name: str, artist_name: Optional[str] = None) -> Optional[str]:
        """Look up a track_id by (approximate) name, using the metadata table."""
        if self.metadata is None:
            raise ValueError("No metadata available to search by name.")

        matches = self.metadata[
            self.metadata["track_name"].str.contains(track_name, case=False, na=False)
        ]
        if artist_name:
            matches = matches[
                matches["artist_name"].str.contains(artist_name, case=False, na=False)
            ]
        return matches.index[0] if len(matches) else None

    def recommend_similar_tracks(
        self,
        seed_track_id: str,
        candidate_track_ids: Optional[List[str]] = None,
        top_k: int = 10,
    ) -> pd.Series:
        """
        Rank candidate tracks by audio-feature similarity to a seed track.
        If candidate_track_ids is None, searches the full loaded dataset.
        """
        if seed_track_id not in self.features.index:
            raise ValueError(f"Unknown seed_track_id: {seed_track_id}")

        seed_vector = self.features.loc[[seed_track_id]]

        if candidate_track_ids is None:
            candidates = self.features.drop(index=seed_track_id)
        else:
            ids = [t for t in candidate_track_ids if t != seed_track_id]
            candidates = self.features.loc[self.features.index.intersection(ids)]

        if candidates.empty:
            return pd.Series(dtype=float)

        sims = cosine_similarity(seed_vector.values, candidates.values)[0]
        return (
            pd.Series(sims, index=candidates.index)
            .sort_values(ascending=False)
            .head(top_k)
        )

    @staticmethod
    def get_live_audio_features(sp, track_ids: List[str]) -> pd.DataFrame:
        """
        Fetch audio features from the live Spotify Web API. Only works for
        apps with grandfathered extended-mode access from before the
        Nov 27, 2024 deprecation — will raise a 403 SpotifyException for
        any new app.

        :param sp: an authenticated spotipy.Spotify client
        """
        features = []
        for i in range(0, len(track_ids), 100):  # API cap: 100 ids per call
            batch = track_ids[i : i + 100]
            results = sp.audio_features(batch)
            features.extend([f for f in results if f is not None])

        df = pd.DataFrame(features)
        if df.empty:
            return df

        df = df.set_index("id")[AUDIO_FEATURE_COLUMNS].copy()
        tempo_range = df["tempo"].max() - df["tempo"].min()
        df["tempo"] = (df["tempo"] - df["tempo"].min()) / (tempo_range + 1e-9)
        return df


# ---------------------------------------------------------------------------
# Hybrid Recommender
# ---------------------------------------------------------------------------


class HybridRecommender:
    """
    Blends collaborative-filtering artist affinity with content-based
    track-level audio-feature similarity.
    """

    def __init__(self, cf: CollaborativeFilteringRecommender, cb: ContentBasedRecommender):
        self.cf = cf
        self.cb = cb

    def recommend(
        self,
        user_id,
        seed_track_id: str,
        candidate_track_ids: Optional[List[str]] = None,
        candidate_artists: Optional[Dict[str, str]] = None,
        cf_weight: float = 0.4,
        top_k: int = 10,
    ) -> pd.Series:
        """
        Rank candidate tracks using a weighted blend of:
          - content-based similarity to the seed track (audio features)
          - a boost if the track's artist is one the CF model would
            recommend for this user

        :param candidate_track_ids: tracks to rank; if None, searches the
            full content-based dataset.
        :param candidate_artists: mapping of track_id -> artist name, used
            to apply the CF boost. If omitted, falls back to the
            content-based recommender's loaded metadata (if available).
        """
        pool_size = len(candidate_track_ids) if candidate_track_ids is not None else 500
        cb_scores = self.cb.recommend_similar_tracks(
            seed_track_id, candidate_track_ids, top_k=pool_size
        )
        if cb_scores.empty:
            return cb_scores

        if candidate_artists is None:
            if self.cb.metadata is None or "artist_name" not in self.cb.metadata.columns:
                raise ValueError(
                    "candidate_artists not provided and no artist metadata "
                    "available on the content-based recommender."
                )
            candidate_artists = self.cb.metadata["artist_name"].to_dict()

        cf_artists = set(self.cf.recommend_for_user(user_id, top_k=50))

        def blended_score(track_id: str, cb_score: float) -> float:
            artist = candidate_artists.get(track_id)
            cf_boost = 1.0 if artist in cf_artists else 0.0
            return (1 - cf_weight) * cb_score + cf_weight * cf_boost

        blended = pd.Series(
            {
                tid: blended_score(tid, score)
                for tid, score in cb_scores.items()
                if tid in candidate_artists
            }
        )
        return blended.sort_values(ascending=False).head(top_k)


# ---------------------------------------------------------------------------
# Offline Evaluation
# ---------------------------------------------------------------------------


def evaluate_leave_one_out(
    listening_matrix: pd.DataFrame,
    n_neighbors: int = 10,
    top_k: int = 10,
    n_users: Optional[int] = 200,
    seed: int = 42,
) -> Dict[str, float]:
    """
    Leave-one-out offline evaluation of the collaborative-filtering
    recommender: for each sampled user, hide one artist they listen to and
    check whether the recommender surfaces it back in the top-k list built
    from their remaining listening history.

    Returns hit-rate@k, precision@k, and recall@k averaged across users.
    This is a real, reproducible substitute for an unverified "satisfaction
    rate" claim.
    """
    rng = np.random.default_rng(seed)
    users = listening_matrix.index.to_numpy()
    if n_users is not None and n_users < len(users):
        users = rng.choice(users, size=n_users, replace=False)

    hits = 0
    evaluated = 0

    for user_id in users:
        listened = listening_matrix.loc[user_id]
        listened_artists = listened[listened == 1].index.tolist()
        if len(listened_artists) < 2:
            continue  # need at least one artist to hide and one to train on

        held_out = rng.choice(listened_artists)

        train_matrix = listening_matrix.copy()
        train_matrix.loc[user_id, held_out] = 0

        cf = CollaborativeFilteringRecommender(train_matrix)
        recs = cf.recommend_for_user(user_id, n_neighbors=n_neighbors, top_k=top_k)

        hits += int(held_out in recs)
        evaluated += 1

    hit_rate = hits / evaluated if evaluated else 0.0
    return {
        "n_users_evaluated": evaluated,
        "hit_rate@k": hit_rate,
        # with exactly one relevant item held out per user, precision@k and
        # recall@k reduce to hits/k and hits/1 respectively, averaged
        "precision@k": hit_rate / top_k,
        "recall@k": hit_rate,
    }
