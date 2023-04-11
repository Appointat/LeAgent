# WIHSPER API
import gradio as gr
import openai

openai.api_key = "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"

def transcribe(audio):
	print(audio)

	audio_file= open(audio, "rb")
	transcript = openai.Audio.transcribe("whisper-1", audio_file)		
	return transcript["text"]

ui = gr.Interface(
    fn=transcribe,
    inputs=gr.Audio(source="microphone", type="filepath"),
    outputs="text"
).launch()

ui.launch()