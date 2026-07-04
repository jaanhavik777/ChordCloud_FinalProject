import os
from typing import List

import spotipy
import streamlit as st
from dotenv import load_dotenv
from spotipy.oauth2 import SpotifyOAuth

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.agent import ReActAgent
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.tools import FunctionTool, QueryEngineTool
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.groq import Groq

load_dotenv()

REQUIRED_ENV_VARS = [
    "SPOTIPY_CLIENT_ID",
    "SPOTIPY_CLIENT_SECRET",
    "SPOTIPY_REDIRECT_URI",
    "GROQ_API_KEY",
]


def check_env_vars() -> List[str]:
    """Return a list of missing required environment variables."""
    return [var for var in REQUIRED_ENV_VARS if not os.getenv(var)]


def get_spotify_client() -> spotipy.Spotify:
    """Create an authenticated Spotify client."""
    return spotipy.Spotify(
        auth_manager=SpotifyOAuth(
            client_id=os.getenv("SPOTIPY_CLIENT_ID"),
            client_secret=os.getenv("SPOTIPY_CLIENT_SECRET"),
            redirect_uri=os.getenv("SPOTIPY_REDIRECT_URI"),
            scope="playlist-modify-private",
        )
    )


def create_spotify_playlist(
    sp: spotipy.Spotify, user_id: str, playlist_name: str, song_list: List[str]
) -> str:
    """
    Creates a Spotify playlist and adds the matched songs to it.

    :param sp: authenticated Spotify client
    :param user_id: Spotify user ID
    :param playlist_name: Name of the playlist to be created
    :param song_list: List of song names (and artists) to search for and add
    :return: A status message including the playlist URL and any songs that
             could not be matched
    """
    playlist = sp.user_playlist_create(user_id, playlist_name, public=False)
    playlist_id = playlist["id"]

    track_uris = []
    not_found = []
    for song in song_list:
        song = song.strip()
        if not song:
            continue
        search_results = sp.search(q=song, type="track", limit=1)
        items = search_results["tracks"]["items"]
        if items:
            track_uris.append(items[0]["uri"])
        else:
            not_found.append(song)

    if track_uris:
        sp.playlist_add_items(playlist_id, track_uris)

    playlist_url = f"https://open.spotify.com/playlist/{playlist_id}"
    message = f"Your playlist has been created! [Open Playlist]({playlist_url})"
    if not_found:
        message += "\n\nCouldn't find a match for:\n" + "\n".join(
            f"- {s}" for s in not_found
        )
    return message


class PlaylistGeneratorWithLlamaIndex:
    def __init__(self, data_path: str):
        """Initialize with LlamaIndex and Spotify API."""
        self.data_path = data_path
        self.sp = get_spotify_client()
        self.index = self.create_index()
        self.song_tool = None  # set inside create_agent
        self.agent = self.create_agent()

    def create_index(self):
        """Create a new index from documents (no persistence)."""
        documents = SimpleDirectoryReader(self.data_path).load_data()
        if not documents:
            raise ValueError("No documents found in the specified path.")
        return VectorStoreIndex.from_documents(documents)

    def create_agent(self) -> ReActAgent:
        """Create an agent with the ability to query the index and search Spotify."""
        llm = Groq(model="llama-3.3-70b-versatile", api_key=os.getenv("GROQ_API_KEY"))

        # Groq doesn't serve embeddings, so use a local HuggingFace model instead.
        embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

        query_engine = self.index.as_query_engine(
            llm=llm, embed_model=embed_model, similarity_top_k=3
        )

        search_tool = QueryEngineTool(
            query_engine=query_engine,
            metadata={
                "name": "document_search",
                "description": "Search through a document corpus to refine song recommendations",
            },
        )

        def song_recommendation_function(query: str) -> str:
            """Fetch Spotify recommendations based on a filtered query string."""
            search_results = self.sp.search(q=query, type="track", limit=5)
            tracks = search_results["tracks"]["items"]
            if not tracks:
                return "No song recommendations found."
            recommendations = [
                f"{track['name']} by {track['artists'][0]['name']}" for track in tracks
            ]
            return "\n".join(recommendations)

        self.song_tool = FunctionTool.from_defaults(
            fn=song_recommendation_function,
            name="song_recommendations",
            description="Fetches song recommendations from Spotify based on a filtered query",
        )

        return ReActAgent.from_tools(
            [search_tool, self.song_tool],
            llm=llm,
            verbose=True,
            memory=ChatMemoryBuffer.from_defaults(token_limit=4096),
        )

    def generate_playlist(self, user_query: str, user_id: str) -> str:
        """Generate a playlist based on the user's query and create it on Spotify."""
        prompt = (
            f"{user_query}\n\n"
            "Use the document_search tool to understand relevant themes/genres, "
            "then use the song_recommendations tool to fetch actual matching tracks. "
            "Finish by returning ONLY the final list of songs, one per line, "
            "in the format 'Song Title by Artist'. Do not include any other commentary."
        )
        response = self.agent.chat(prompt)

        song_list = [
            line.strip("-* \t")
            for line in str(response).splitlines()
            if line.strip()
        ]

        if not song_list:
            raise ValueError("The agent did not return any song recommendations.")

        return create_spotify_playlist(self.sp, user_id, "Generated Playlist", song_list)


@st.cache_resource(show_spinner="Loading music library and model...")
def get_playlist_generator(data_path: str) -> PlaylistGeneratorWithLlamaIndex:
    return PlaylistGeneratorWithLlamaIndex(data_path=data_path)


def main():
    st.title("ChordCloud's Personalized Playlist Generator")

    missing = check_env_vars()
    if missing:
        st.error(
            "Missing required environment variable(s): "
            + ", ".join(missing)
            + ". Add them to your .env file before continuing."
        )
        return

    user_id = st.text_input("Enter your Spotify user ID:")
    user_query = st.text_area(
        "Describe the playlist you want (e.g., 'Relaxing music for studying'):"
    )

    if st.button("Generate Playlist"):
        if not user_query or not user_id:
            st.error("Please enter both a Spotify user ID and a playlist query.")
            return

        data_path = "data/music_data"
        if not os.path.isdir(data_path):
            st.error(f"Data path '{data_path}' not found. Add your music docs there first.")
            return

        try:
            with st.spinner("Generating your playlist..."):
                playlist_generator = get_playlist_generator(data_path)
                result_message = playlist_generator.generate_playlist(user_query, user_id)
            st.success("Your personalized playlist has been generated!")
            st.markdown(result_message)
        except Exception as e:
            st.error(f"Error: {e}")


if __name__ == "__main__":
    main()
