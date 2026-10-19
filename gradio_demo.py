import gradio as gr
from huggingface_hub import HfFolder

def add_numbers(Num1, Num2):
    return Num1 + Num2

def add_sentences(text1, text2):
    return f"{text1} {text2}"

# Define the interface
demo = gr.Interface(
    fn=add_sentences,
    inputs=[gr.Textbox(label="Input 1"), gr.Textbox(label="Input 1")], # Create two numerical input fields where users can enter numbers
    outputs=gr.Textbox(label="Output")  # Create numerical output fields
)

# Launch the interface
demo.launch(server_name="127.0.0.1", server_port= 7860)
