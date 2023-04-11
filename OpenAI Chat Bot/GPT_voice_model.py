# -*- coding: utf-8 -*-
from argparse import ArgumentDefaultsHelpFormatter
import gradio as gr
import os
import openai, subprocess

openai.api_key = "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"

messages = [{"role": "system", "content": 'YOUR ARE A BOT.'}]

def rename_file(audio):
    new_file_name = os.path.splitext(audio)[0] + '.wav'
    os.rename(audio, new_file_name)
    return new_file_name


def transcribe(audio):
    global messages
    audio = rename_file(audio)

    audio_file = open(audio, "rb")
    transcript = openai.Audio.transcribe("whisper-1", audio_file)


    messages.append({"role": "user", "content": transcript["text"]})

    response = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=messages)

    system_message = response["choices"][0]["message"]
    messages.append(system_message)

    subprocess.call(["wsay", system_message['content']])

    chat_transcript = ""
    for message in messages:
        if message['role'] != 'system':
            chat_transcript += message['role'] + ": " + message['content'] + "\n\n"

    return chat_transcript

ui = gr.Interface(fn = transcribe, 
                  inputs = gr.Audio(source = "microphone", type = "filepath"), 
                  outputs = "text"
                  ).launch()

ui.launch()