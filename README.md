# 🤥 cyrano 👺

** DISCLAMER: I'm not great at writing documentation, feel free to open an issue if you need more support on how to use **

Your personal assistant. Cyrano is a large language model which has access to a set of tools it can choose to use whenever they are needed to answer the user. Its name comes from the French theater character [Cyrano de Bergerac](https://fr.wikipedia.org/wiki/Cyrano_de_Bergerac_(Rostand)) by Edmond Rostand.

Cyrano uses the last OpenAI models: gpt-3.5-turbo-0613 (faster) or gpt-4-0613 (smarter). They have access to [OpenAI functions](https://openai.com/blog/function-calling-and-other-api-updates). The repo is based on the [langchain](https://python.langchain.com/docs/get_started/introduction.html) framework.

## demo
TODO : add examples and/or a video

## Base Tools
- Search *(Google search)*
    - live weather
    - recent news
    - more reliable general information
    - ...
- Python *(write and execute code)*
    - solve mathematical questions
- Timer *(launch a timer)*
    - the timer runs a separate thread so that you can continue to interact with the model while it waits

## Chess
Cyrano uses [python-chess](https://python-chess.readthedocs.io/en/latest/) to keep an internal chess board state at all time. When hearing a chess move in natural language in the conversation, it translates it to the corresponding algebraic representation and play it on its internal python-chess Board. It uses [stockfish](https://stockfishchess.org/) for 0.1 second to decide what to play in response. The board is then saved as a [FEN](https://en.wikipedia.org/wiki/Forsyth%E2%80%93Edwards_Notation) in a json file so that you can resume your game at any time in the future.

Cyrano has access to 3 chess related actions:
- **PlayMove**: update the board with a user move and a stockfish move
- **GetBoardState**: return the board FEN and the last_move_date date and time 
- **ResetBoard**: reset the board state for a new game

## Memory

### Short Term Memory (STM)
Cyrano stores the last messages up to a certain amount of tokens (default 1500) in its context window.

### Long Term Memory (LTM)
*based on the paper [Generative Agents: Interactive Simulacra of Human Behavior](https://arxiv.org/abs/2304.03442) by Park, et. al.*

Long Term Memory uses a time-weighted vector store. Whenever the conversation exceeds 1500 tokens, the oldest messages are summarized, vectorized (using a sentence transformers model) and stored in a vector store database (Qdrant). Each time Cyrano stores a memory, it scores them out of 10 on how memorable they are. Each memory is a short text, the corresponding vector, the importance_score, a created_at datetime object, and a last_accessed_at datetime object. 

At runtime, Cyrano fetches the top 5 most relevant memories and adds them to the last user message. When choosing the most relevant memories, it uses semantic similarity, weighted by the age of the memory, using the formula: ` semantic_similarity + (1 - decay_rate) ** hours_passed`. For which semantic_similarity is the cosine distance between vectors, decay_rate represents how quickly memories lose relevance, and hours_passed is the delta between the present and last_accessed_at.

## Sound

### Cyrano as text only

you can use Cyrano as a text only agent by setting the sound variable to False in src/main.py. I you choose to do so, inputs will be python `input()` and outputs will be texts.

### 1. Wake word

If you choose to interact with it using voice and ear, you will need to use [porcupine](https://picovoice.ai/docs/porcupine/) for the wake word detection. For this "alexa" or "hey google" equivalent, I simply chose "Cyrano". Whenever you want to interact with the model you can say its name out loud, wait for a beep, and start talking. 

NOTE: The wake word is currently set with a French accent. simply regenerate it from porcupine website in another language if needed.

### 2. Speech to Text

Sound is recorded using the [SpeechRecognition](https://pypi.org/project/SpeechRecognition/) library. after hearing a silence, it will stop recording and send the sound file to [OpenAI Whisper](https://openai.com/research/whisper) API which will send back the transcribed text.

### 3. Text To Speech (TTS)

Cyrano can use 3 TTS options:

- [pyttsx3](https://pypi.org/project/pyttsx3/) which is free and have a robotic but acceptable voice on windows.

- [Google cloud TTS](https://cloud.google.com/text-to-speech?hl=fr) very good quality/ price option imo. Works well on any os.

- [Eleven labs](https://elevenlabs.io/) best sound quality, but the free version is limited and the paid version do not offer enough character per month.

As I needed Cyrano to run on a Raspberry, wanted good quality, and high volume, so not too pricey, I chose Google TTS. I left all functions for the 3 options in src/sound_utils.py. Feel free to use the one which suits you best. for google or eleven labs, you will need to add the api key in the .env file.


## installation

### 1. create a .env file

clone the repo, then create a .env file with all api_key and a system prompt containing the following:
- example of system prompt (the identity of the model):

    ```SYS_PROMPT="You are Cyrano, a personal assistant with the personality of Cyrano de Bergerac. Today's date is {current_date}. You're on {user_name}'s desk. {user_description}. {user_name}'s messages are recorded in sound and then transcribed into text. It may happen that the sound is incorrectly transcribed. You regularly reply in a sarcastic and humorous manner."```

- [OpenAI api key](https://platform.openai.com/docs/api-reference/authentication)

- [Google search API key](https://serper.dev/)

- OPTIONNAL: [Google search API key](https://serper.dev/)


### Use directly (recommended)

install the required libraries: `pip install -r requirements.txt`

launch the app: `python src/main.py`

when starting the app should play a few notes. Say "cyrano" when you want to interact with it, a beep will notify that it's listening and another when it detects a silence and stops transcribing.

### Use Docker (unstable)
- docker build
- docker run

### backlog:
- make launching spotify works again
- update timer alarm with better sound
