# ChordCloud — AI-Powered Music Recommendation System

Two related but independent pieces:

1. **`app.py`** — a Streamlit app that uses a LlamaIndex RAG agent (Groq LLM +
   local embeddings) to interpret natural-language playlist requests, search
   a document corpus for relevant themes, and generate a matching Spotify
   playlist via the Spotify Web API.
2. **`music_recommender.py`** — a standalone recommendation engine
   demonstrating collaborative filtering, content-based filtering, and a
   hybrid approach, with real offline evaluation. Decoupled from the app so
   it can be run, tested, and cited on its own. See
   `music_recommendation_analysis.ipynb` for a runnable walkthrough.

## Setup

```bash
# For the Streamlit app
pip install -r requirements.txt

# For the recommender module / notebook
pip install -r requirements-recommender.txt
```

Create a `.env` file (for `app.py`) with:
```
SPOTIPY_CLIENT_ID=...
SPOTIPY_CLIENT_SECRET=...
SPOTIPY_REDIRECT_URI=...
GROQ_API_KEY=...
```

## Running the app

```bash
streamlit run app.py
```

Add documents describing your music corpus under `data/music_data/` first
(the app raises a clear error if that folder is missing or empty).

## Running the recommender notebook

```bash
jupyter notebook music_recommendation_analysis.ipynb
```

Uses the datasets in `data/`:
- `lastfm-matrix-germany.csv` — public user-artist listening matrix (1,257
  users × 285 artists), used for collaborative filtering.
- `SpotifyFeatures.csv` — public precomputed Spotify audio-features dataset
  (176,774 unique tracks, 27 genres), used for content-based filtering.

## Running the tests

```bash
pytest test_music_recommender.py -v
```

## A note on the Spotify audio-features data source

Spotify deprecated the live `audio_features` / `audio_analysis` Web API
endpoints for new third-party apps on November 27, 2024 — new apps get a 403
with no official replacement. Because of that, `ContentBasedRecommender`
loads audio features from the public precomputed dataset above rather than
calling the live endpoint. A `get_live_audio_features()` static method is
still included for anyone whose app has grandfathered extended-mode access.

## Real, reproducible numbers (not invented stats)

Re-run `evaluate_leave_one_out()` in the notebook any time the model or
dataset changes — as of this build, on a 300-user sample:

```
hit_rate@10:  0.254
precision@10: 0.025
recall@10:    0.254
```

## Project structure

```
.
├── app.py                                # Streamlit playlist-generator app
├── requirements.txt                      # deps for app.py
├── music_recommender.py                  # CF / content-based / hybrid recommender + eval
├── requirements-recommender.txt          # deps for music_recommender.py / notebook
├── test_music_recommender.py             # pytest unit tests
├── music_recommendation_analysis.ipynb   # runnable demo + evaluation
├── data/
│   ├── lastfm-matrix-germany.csv         # CF dataset
│   └── SpotifyFeatures.csv               # content-based dataset
└── README.md
```
