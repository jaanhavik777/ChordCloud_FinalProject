import os
import random
import streamlit as st
from dotenv import load_dotenv
import requests
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials, SpotifyOAuth
from typing import List, Tuple, Dict

# Load environment variables
load_dotenv()

# Retrieve API keys and credentials
serperdev_api_key = os.getenv("SERPERDEV_API_KEY")
groq_api_key = os.getenv("GROQ_API_KEY")
spotipy_client_id = os.getenv("SPOTIPY_CLIENT_ID")
spotipy_client_secret = os.getenv("SPOTIPY_CLIENT_SECRET")

# Set up Spotify client
spotify = spotipy.Spotify(
    client_credentials_manager=SpotifyClientCredentials(
        client_id=spotipy_client_id,
        client_secret=spotipy_client_secret
    )
)

# Function to fetch playlist data from Serper API
def fetch_playlist_data(query: str) -> Dict:
    headers = {
        'Authorization': f'Bearer {serperdev_api_key}',
        'Content-Type': 'application/json'
    }
    try:
        response = requests.get(
            f'https://google.serper.dev/search?q=&apiKey=3985e8024153834462bacff357bacea200546eff',
            params={'q': query},
            headers=headers,
            timeout=10
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching data from Serper: {e}")
        return {'organic': []}

# Function to create playlist based on user query
def create_playlist(query: str) -> Tuple[str, List[str]]:
    search_results = fetch_playlist_data(query)
    relevant_songs = [result['title'] for result in search_results.get('organic', []) if 'title' in result]

    if not relevant_songs:
        return "No songs found.", []
    
    return relevant_songs

# Function to recommend similar songs using Spotify
def recommend_similar_songs(song_title: str, num_recommendations: int) -> List[str]:
    try:
        search_results = spotify.search(q=song_title, type='track', limit=1)
        tracks = search_results['tracks']['items']
        
        if not tracks:
            return ["No similar songs found."]
        
        track_id = tracks[0]['id']
        recommendations = spotify.recommendations(seed_tracks=[track_id], limit=num_recommendations)
        similar_songs = [f"{track['name']} by {track['artists'][0]['name']}" 
                         for track in recommendations['tracks']]
        
        return similar_songs
    except Exception as e:
        st.error(f"Error getting recommendations: {e}")
        return ["Error fetching recommendations."]
        
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
    st.title("ChordCloud's Personalized Playlist Recommender")

    # Input for user's Spotify username
    user_id = st.text_input("Enter your Spotify user ID:")

    # Input for playlist query
    user_query = st.text_area("Describe the playlist you want (e.g., 'Relaxing music for studying'):") 
    
    if user_query:
        relevant_songs = create_playlist(user_query)

        if relevant_songs:
            # Let the user choose how many recommendations to generate
            num_recommendations = st.slider("How many songs would you like?", min_value=1, max_value=30, value=5)
            # Generate similar songs based on the first song in the playlist
            similar_songs = recommend_similar_songs(relevant_songs[0], num_recommendations)

            st.subheader("Recommended Songs:")
            for i, song in enumerate(similar_songs, 1):
                st.write(f"{i}. {song}")

            # Spotify authentication
            REDIRECT_URI = "https://chordcloud/callback"
            scope = "playlist-modify-private"
            sp = spotipy.Spotify(auth_manager=SpotifyOAuth(client_id=spotipy_client_id,
                                                           client_secret=spotipy_client_secret,
                                                           redirect_uri=REDIRECT_URI,
                                                           scope=scope))

        


# Run the Streamlit app
if __name__ == "__main__":
    main()