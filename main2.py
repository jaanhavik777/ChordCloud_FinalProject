import os
import spotipy
from spotipy.oauth2 import SpotifyOAuth
from dotenv import load_dotenv
from typing import List
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, ToolMetadata
from llama_index.llms.groq import Groq
from llama_index.embeddings.jinaai import JinaEmbedding
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
    def __init__(self, data_path: str, index_path: str = "index"):
        """Initialize with LlamaIndex and Spotify API."""
        self.data_path = data_path
        self.index_path = index_path
        self.sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(
            client_id=os.getenv("SPOTIPY_CLIENT_ID"),
            client_secret=os.getenv("SPOTIPY_CLIENT_SECRET")
        ))

        # Load or create the index with LlamaIndex
        self.index = self.load_or_create_index()
        
        # Set up agent for querying the index
        self.agent = self.create_agent()

    def load_or_create_index(self):
        """Load existing index or create new one."""
        if os.path.exists(self.index_path):
            return self.load_index()
        else:
            return self.create_index()

    def load_index(self):
        """Load existing vector index."""
        storage_context = StorageContext.from_defaults(persist_dir=self.index_path)
        return load_index_from_storage(storage_context)

    def create_index(self):
        """Create a new index from documents."""
        documents = SimpleDirectoryReader(self.data_path).load_data()
        if not documents:
            raise ValueError("No documents found in the specified path.")
        return VectorStoreIndex.from_documents(documents)

    def create_agent(self):
        """Create an agent with the ability to query the index."""
        query_engine = self.index.as_query_engine(similarity_top_k=3)

        # Basic search tool
        search_tool = QueryEngineTool(
            query_engine=query_engine,
            metadata=ToolMetadata(
                name="document_search",
                description="Search through a document corpus to refine song recommendations",
            ),
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


# Example usage
if __name__ == "__main__":
    data_path = "./your_music_data" 

    # Instantiate the playlist generator
    playlist_generator = PlaylistGeneratorWithLlamaIndex(data_path=data_path)

    # User query for playlist generation
    user_query = "Create a playlist for a relaxing afternoon"
    
    # Spotify user ID (this can be fetched via the Spotify API if needed)
    user_id = "your_spotify_user_id"  # Replace with the actual Spotify user ID

    # Generate and display the playlist
    playlist_url = playlist_generator.generate_playlist(user_query, user_id)
    print(playlist_url)
