import gradio as gr

def transcribe(audio):
	print(audio)
	return "Audio here."

ui = gr.Interface(
	fn = transcribe,
	inputs = gr.Audio(source = "microphone", path = "filepath"),
	outputs = "text",
).launch()

ui.launch()