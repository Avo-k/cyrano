from dotenv import load_dotenv
import os
import spotipy
import webbrowser


# LOAD API KEYS
load_dotenv()
clientID = os.getenv("SPOTIFY_CLIENT_ID")
clientSecret = os.getenv("SPOTIFY_CLIENT_SECRET")
redirect_uri = "http://google.com/callback/"

oauth_object = spotipy.SpotifyOAuth(clientID, clientSecret, redirect_uri)
token_dict = oauth_object.get_access_token()
token = token_dict["access_token"]
spotifyObject = spotipy.Spotify(auth=token)
user_name = spotifyObject.current_user()


def play_music(song_name: str) -> str:
    results = spotifyObject.search(song_name, 1, 0, "track")
    song_items = results["tracks"]["items"]
    song = song_items[0]["external_urls"]["spotify"]
    webbrowser.open(song)
    return f"playing {song_items[0]['name']} by {song_items[0]['artists'][0]['name']}"
