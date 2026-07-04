"""
Unit tests for music_recommender.py.

Run with:
    pytest test_music_recommender.py -v
"""

import numpy as np
import pandas as pd
import pytest

from music_recommender import (
    CollaborativeFilteringRecommender,
    ContentBasedRecommender,
    HybridRecommender,
    evaluate_leave_one_out,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def listening_matrix():
    """Small synthetic binary user-artist listening matrix."""
    data = {
        "queen":     [1, 1, 0, 0, 1],
        "abba":      [1, 0, 0, 0, 1],
        "metallica": [0, 1, 1, 1, 0],
        "megadeth":  [0, 1, 1, 1, 0],
        "coldplay":  [1, 0, 0, 1, 1],
    }
    index = ["u1", "u2", "u3", "u4", "u5"]
    return pd.DataFrame(data, index=index)


@pytest.fixture
def audio_features_csv(tmp_path):
    """Small synthetic Spotify-style audio-features CSV."""
    rows = [
        {"track_id": "t1", "track_name": "Song A", "artist_name": "Artist X", "genre": "Rock",
         "danceability": 0.8, "energy": 0.9, "tempo": 140, "valence": 0.7, "acousticness": 0.1,
         "instrumentalness": 0.0, "speechiness": 0.05, "liveness": 0.1},
        {"track_id": "t2", "track_name": "Song B", "artist_name": "Artist Y", "genre": "Rock",
         "danceability": 0.75, "energy": 0.85, "tempo": 138, "valence": 0.65, "acousticness": 0.15,
         "instrumentalness": 0.0, "speechiness": 0.06, "liveness": 0.12},
        {"track_id": "t3", "track_name": "Song C", "artist_name": "Artist Z", "genre": "Jazz",
         "danceability": 0.2, "energy": 0.1, "tempo": 70, "valence": 0.3, "acousticness": 0.9,
         "instrumentalness": 0.8, "speechiness": 0.02, "liveness": 0.3},
        {"track_id": "t1_dup", "track_name": "Song A", "artist_name": "Artist X", "genre": "Alt-Rock",
         "danceability": 0.8, "energy": 0.9, "tempo": 140, "valence": 0.7, "acousticness": 0.1,
         "instrumentalness": 0.0, "speechiness": 0.05, "liveness": 0.1},
    ]
    # duplicate track_id "t1" appears twice under different genres, to exercise
    # the drop_duplicates(subset="track_id") path -- rename for clarity
    rows[3]["track_id"] = "t1"

    df = pd.DataFrame(rows)
    path = tmp_path / "features.csv"
    df.to_csv(path, index=False)
    return path


# ---------------------------------------------------------------------------
# CollaborativeFilteringRecommender
# ---------------------------------------------------------------------------


class TestCollaborativeFilteringRecommender:
    def test_recommend_for_user_excludes_already_listened(self, listening_matrix):
        cf = CollaborativeFilteringRecommender(listening_matrix)
        recs = cf.recommend_for_user("u1", n_neighbors=4, top_k=5)
        already_listened = set(listening_matrix.loc["u1"][listening_matrix.loc["u1"] == 1].index)
        assert not (set(recs) & already_listened)

    def test_recommend_for_user_returns_known_artists(self, listening_matrix):
        cf = CollaborativeFilteringRecommender(listening_matrix)
        recs = cf.recommend_for_user("u1", top_k=5)
        assert set(recs).issubset(set(listening_matrix.columns))

    def test_recommend_for_user_unknown_user_raises(self, listening_matrix):
        cf = CollaborativeFilteringRecommender(listening_matrix)
        with pytest.raises(KeyError):
            cf.recommend_for_user("nonexistent_user")

    def test_recommend_similar_artists_excludes_self(self, listening_matrix):
        cf = CollaborativeFilteringRecommender(listening_matrix)
        sims = cf.recommend_similar_artists("metallica", top_k=4)
        assert "metallica" not in sims

    def test_recommend_similar_artists_finds_co_listened(self, listening_matrix):
        # metallica and megadeth are listened to by exactly the same users
        # in the fixture, so they should be perfectly similar to each other
        cf = CollaborativeFilteringRecommender(listening_matrix)
        sims = cf.recommend_similar_artists("metallica", top_k=1)
        assert sims[0] == "megadeth"

    def test_recommend_similar_artists_unknown_artist_raises(self, listening_matrix):
        cf = CollaborativeFilteringRecommender(listening_matrix)
        with pytest.raises(KeyError):
            cf.recommend_similar_artists("nonexistent artist")

    def test_user_similarity_matrix_is_symmetric(self, listening_matrix):
        cf = CollaborativeFilteringRecommender(listening_matrix)
        sim = cf.user_similarity
        np.testing.assert_allclose(sim.values, sim.values.T, atol=1e-8)


# ---------------------------------------------------------------------------
# ContentBasedRecommender
# ---------------------------------------------------------------------------


class TestContentBasedRecommender:
    def test_from_csv_dedupes_by_track_id(self, audio_features_csv):
        cb = ContentBasedRecommender.from_csv(str(audio_features_csv))
        assert cb.features.index.is_unique
        assert cb.features.shape[0] == 3  # t1, t2, t3 (t1 duplicate collapsed)

    def test_from_csv_normalizes_tempo_to_unit_range(self, audio_features_csv):
        cb = ContentBasedRecommender.from_csv(str(audio_features_csv))
        assert cb.features["tempo"].max() <= 1.0
        assert cb.features["tempo"].min() >= 0.0

    def test_find_track_id_matches_by_name(self, audio_features_csv):
        cb = ContentBasedRecommender.from_csv(str(audio_features_csv))
        found = cb.find_track_id("Song B")
        assert found == "t2"

    def test_find_track_id_filters_by_artist(self, audio_features_csv):
        cb = ContentBasedRecommender.from_csv(str(audio_features_csv))
        found = cb.find_track_id("Song A", artist_name="Artist X")
        assert found == "t1"

    def test_find_track_id_no_match_returns_none(self, audio_features_csv):
        cb = ContentBasedRecommender.from_csv(str(audio_features_csv))
        assert cb.find_track_id("Nonexistent Track Title") is None

    def test_recommend_similar_tracks_ranks_closer_track_first(self, audio_features_csv):
        cb = ContentBasedRecommender.from_csv(str(audio_features_csv))
        # t1 and t2 are acoustically close (both energetic rock); t3 is a
        # quiet acoustic jazz track, so t2 should rank above t3 as similar to t1
        sims = cb.recommend_similar_tracks("t1", top_k=2)
        assert list(sims.index)[0] == "t2"

    def test_recommend_similar_tracks_unknown_seed_raises(self, audio_features_csv):
        cb = ContentBasedRecommender.from_csv(str(audio_features_csv))
        with pytest.raises(ValueError):
            cb.recommend_similar_tracks("nonexistent_track_id")

    def test_recommend_similar_tracks_restricts_to_candidate_pool(self, audio_features_csv):
        cb = ContentBasedRecommender.from_csv(str(audio_features_csv))
        sims = cb.recommend_similar_tracks("t1", candidate_track_ids=["t3"], top_k=5)
        assert list(sims.index) == ["t3"]


# ---------------------------------------------------------------------------
# HybridRecommender
# ---------------------------------------------------------------------------


class TestHybridRecommender:
    def test_recommend_with_explicit_candidate_artists(self, listening_matrix, audio_features_csv):
        cf = CollaborativeFilteringRecommender(listening_matrix)
        cb = ContentBasedRecommender.from_csv(str(audio_features_csv))
        hybrid = HybridRecommender(cf, cb)

        candidate_artists = {"t2": "queen", "t3": "some other artist"}
        recs = hybrid.recommend(
            user_id="u1",
            seed_track_id="t1",
            candidate_track_ids=["t2", "t3"],
            candidate_artists=candidate_artists,
            cf_weight=0.5,
            top_k=2,
        )
        assert set(recs.index).issubset({"t2", "t3"})

    def test_recommend_falls_back_to_metadata_artists(self, listening_matrix, audio_features_csv):
        cf = CollaborativeFilteringRecommender(listening_matrix)
        cb = ContentBasedRecommender.from_csv(str(audio_features_csv))
        hybrid = HybridRecommender(cf, cb)

        # no candidate_artists passed -> should use cb.metadata automatically
        recs = hybrid.recommend(user_id="u1", seed_track_id="t1", top_k=2)
        assert len(recs) <= 2

    def test_cf_weight_zero_matches_pure_content_based_ranking(self, listening_matrix, audio_features_csv):
        cf = CollaborativeFilteringRecommender(listening_matrix)
        cb = ContentBasedRecommender.from_csv(str(audio_features_csv))
        hybrid = HybridRecommender(cf, cb)

        cb_only = cb.recommend_similar_tracks("t1", candidate_track_ids=["t2", "t3"], top_k=2)
        hybrid_recs = hybrid.recommend(
            user_id="u1",
            seed_track_id="t1",
            candidate_track_ids=["t2", "t3"],
            candidate_artists={"t2": "x", "t3": "y"},
            cf_weight=0.0,
            top_k=2,
        )
        assert list(hybrid_recs.index) == list(cb_only.index)


# ---------------------------------------------------------------------------
# Offline evaluation
# ---------------------------------------------------------------------------


class TestEvaluateLeaveOneOut:
    def test_returns_expected_keys(self, listening_matrix):
        results = evaluate_leave_one_out(listening_matrix, n_neighbors=2, top_k=2, n_users=5)
        assert set(results.keys()) == {
            "n_users_evaluated",
            "hit_rate@k",
            "precision@k",
            "recall@k",
        }

    def test_metrics_are_in_valid_range(self, listening_matrix):
        results = evaluate_leave_one_out(listening_matrix, n_neighbors=2, top_k=2, n_users=5)
        assert 0.0 <= results["hit_rate@k"] <= 1.0
        assert 0.0 <= results["precision@k"] <= 1.0
        assert 0.0 <= results["recall@k"] <= 1.0

    def test_is_reproducible_given_same_seed(self, listening_matrix):
        r1 = evaluate_leave_one_out(listening_matrix, n_users=5, seed=1)
        r2 = evaluate_leave_one_out(listening_matrix, n_users=5, seed=1)
        assert r1 == r2

    def test_skips_users_with_fewer_than_two_listens(self):
        # user with only one listened artist can't have one hidden and one
        # left to train on, so should be excluded from evaluation
        matrix = pd.DataFrame(
            {"a": [1, 0], "b": [0, 1], "c": [0, 1]}, index=["sparse_user", "u2"]
        )
        results = evaluate_leave_one_out(matrix, n_users=2)
        assert results["n_users_evaluated"] == 1
