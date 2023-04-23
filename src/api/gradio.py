import gradio as gr
import logging
import configparser
from pathlib import Path

from src.loader.base_loader import convert_to_samples

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

config = configparser.ConfigParser()
config.read(Path('configs', 'config.ini'))


def predict(**kwargs):
    return {"labels": 0.01, "labels2": 0.99}


with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column() as input_col:
            text1 = gr.Textbox(label="t1")
            slider1 = gr.Slider(1, 10, value=5, step=1)
            number1 = gr.Number()
            checkbox = gr.CheckboxGroup(["Option1", "Option2", "Option3"])
            button = gr.Button("submit")
        with gr.Column() as output_col:
            label = gr.Label(num_top_classes=100)

    button.click(fn=predict, inputs=[slider1, number1, checkbox], outputs=[label])


demo.launch()
