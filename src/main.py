import time


st = time.perf_counter()


from sound_utils import listen_for_wake_word, listen, make_beep, google_tts
from agent_langchain import init_agent, stream_sentences
import os


agent = init_agent()
reponse_directe = False

sound = False

# starting beeps (ba dum)
# make_beep(False)
# make_beep()
print(f"ready :) started in {time.perf_counter() - st:.1f} sec")

while True:
    full = ""

    if sound:
        # use voice to communicate with the agent
        if reponse_directe or listen_for_wake_word():
            user_message = listen()
            for next_sentence in stream_sentences(agent, user_message):
                # print(end=next_sentence)
                google_tts(next_sentence)
                full += next_sentence
            # print(f"Jean: {user_message}")
            # agent_answer = agent.run(user_message)
            # google_tts(agent_answer)
            # reponse_directe = True
        else:
            break

    else:
        # use text to communicate with the agent
        user_message = input()
        # user_message = "quelle ouverture d'échec jean joue t il ?"
        # print("start")
        print()
        for next_sentence in stream_sentences(agent, user_message):
            print(end=next_sentence)
            # google_tts(next_sentence)
            full += next_sentence
        print("\n", "-" * 50)
        # print("full")
        # print(agent.memory.chat_memory.messages)
        # google_tts(full)
        # print(full)
