import os
import spotipy
from spotipy.oauth2 import SpotifyOAuth
import streamlit as st
from dotenv import load_dotenv
from typing import List
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from langchain.llms import OpenAI  # Use OpenAI from langchain
from langchain.embeddings.openai import OpenAIEmbeddings  # Use OpenAI embeddings from langchain
from llama_index.core.agent import ReActAgent
from llama_index.core.tools import QueryEngineTool, FunctionTool
from llama_index.core.memory import ChatMemoryBuffer

# Load environment variables from .env file
load_dotenv()

# Function to create a Spotify playlist with the recommended songs
def create_spotify_playlist(user_id: str, playlist_name: str, song_list: List[str]) -> str:
    """
    Creates a Spotify playlist and adds the song list to it.
    
    :param user_id: Spotify user ID
    :param playlist_name: Name of the playlist to be created
    :param song_list: List of song names (and artists) to be added to the playlist
    :return: The URL of the created playlist
    """
    # Authenticate user for Spotify API access (scope: playlist-modify-private)
    sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
        client_id=os.getenv("SPOTIPY_CLIENT_ID"),
        client_secret=os.getenv("SPOTIPY_CLIENT_SECRET"),
        redirect_uri=os.getenv("SPOTIPY_REDIRECT_URI"),
        scope="playlist-modify-private"
    ))

    # Create a new private playlist
    playlist = sp.user_playlist_create(user_id, playlist_name, public=False)
    playlist_id = playlist['id']

    # Search for each song and get its URI
    track_uris = []
    for song in song_list:
        search_results = sp.search(q=song, type="track", limit=1)
        if search_results['tracks']['items']:
            track_uri = search_results['tracks']['items'][0]['uri']
            track_uris.append(track_uri)

    # Add tracks to the created playlist
    if track_uris:
        sp.playlist_add_items(playlist_id, track_uris)

    # Return the URL of the created playlist
    return f"Your playlist has been created! [Open Playlist](https://open.spotify.com/playlist/{playlist_id})"


class PlaylistGeneratorWithLlamaIndex:
    def __init__(self, data_path: str):
        """Initialize with LlamaIndex and Spotify API."""
        self.data_path = data_path
        self.sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
            client_id=os.getenv("SPOTIPY_CLIENT_ID"),
            client_secret=os.getenv("SPOTIPY_CLIENT_SECRET"),
            redirect_uri=os.getenv("SPOTIPY_REDIRECT_URI"),
            scope="playlist-modify-private"
        ))

        # Load the index with LlamaIndex
        self.index = self.create_index()
        
        # Set up agent for querying the index
        self.agent = self.create_agent()

    def create_index(self):
        """Create a new index from documents (no persistence)."""
        documents = SimpleDirectoryReader(self.data_path).load_data()
        if not documents:
            raise ValueError("No documents found in the specified path.")
        return VectorStoreIndex.from_documents(documents)

    def create_agent(self):
        """Create an agent with the ability to query the index."""
        # Initialize OpenAI as the LLM (You will need to set the OpenAI API key in your .env file)
        llm = OpenAI(model="text-davinci-003", api_key=os.getenv("OPENAI_API_KEY"))
        
        # Use OpenAI for embeddings via langchain's OpenAIEmbeddings
        embed_model = OpenAIEmbeddings(model="text-davinci-003", openai_api_key=os.getenv("OPENAI_API_KEY"))

        
        query_engine = self.index.as_query_engine(llm=llm, embed_model=embed_model, similarity_top_k=3)

        # Basic search tool
        search_tool = QueryEngineTool(
            query_engine=query_engine,
            name="document_search",
            description="Search through a document corpus to refine song recommendations",
        )

        # Custom tool to format song recommendations
        def song_recommendation_function(query: str) -> str:
            """Fetch Spotify recommendations based on filtered query."""
            search_results = self.sp.search(q=query, type="track", limit=5)
            tracks = search_results['tracks']['items']
            
            if not tracks:
                return "No song recommendations found."
            
            recommendations = [f"{track['name']} by {track['artists'][0]['name']}" for track in tracks]
            return "\n".join(recommendations)

        song_recommendation_tool = FunctionTool.from_defaults(
            fn=song_recommendation_function,
            name="song_recommendations",
            description="Fetches song recommendations from Spotify based on the filtered query"
        )

        # Initialize the agent with tools
        return ReActAgent.from_tools(
            [search_tool, song_recommendation_tool],
            verbose=True,
            memory=ChatMemoryBuffer.from_defaults(token_limit=4096),
        )

    def generate_playlist(self, user_query: str, user_id: str) -> str:
        """Generate a playlist based on the user's query and create it on Spotify."""
        response = self.agent.chat(user_query)
        
        # Get the refined query from the agent's response
        refined_query = response.response
        
        # Fetch song recommendations based on the refined query
        playlist_songs = self.agent.tools[1].fn(refined_query)
        
        # Convert the song recommendations into a list
        song_list = playlist_songs.split("\n")
        
        # Create the Spotify playlist
        playlist_url = create_spotify_playlist(user_id, "Generated Playlist", song_list)
        
        return playlist_url


# Streamlit interface
def main():
    st.title("ChordCloud's Personalized Playlist Generator")

    # Input for user's Spotify username
    user_id = st.text_input("Enter your Spotify user ID:")

    # Input for playlist query
    user_query = st.text_area("Describe the playlist you want (e.g., 'Relaxing music for studying'):")

    if st.button("Generate Playlist"):
        if not user_query or not user_id:
            st.error("Please enter both a Spotify user ID and a playlist query.")
            return
        
        # Path to your data directory (you should place your music-related documents here)
        data_path = "data/music_data"  # Adjust this path to your directory containing music-related documents

        # Instantiate the playlist generator
        playlist_generator = PlaylistGeneratorWithLlamaIndex(data_path=data_path)

        try:
            # Generate the playlist and display the link
            playlist_url = playlist_generator.generate_playlist(user_query, user_id)
            st.success("Your personalized playlist has been generated!")
            st.markdown(playlist_url)
        except Exception as e:
            st.error(f"Error: {e}")

if __name__ == "__main__":
    main()
