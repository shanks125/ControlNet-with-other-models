#!/usr/bin/env python

from __future__ import annotations

import os
import pathlib
import shlex
import subprocess

import gradio as gr

if os.getenv('SYSTEM') == 'spaces':
    with open('patch') as f:
        subprocess.run(shlex.split('patch -p1'), stdin=f, cwd='ControlNet')

base_url = 'https://huggingface.co/lllyasviel/ControlNet/resolve/main/annotator/ckpts/'
names = [
    'body_pose_model.pth',
    'dpt_hybrid-midas-501f0c75.pt',
    'hand_pose_model.pth',
    'mlsd_large_512_fp32.pth',
    'mlsd_tiny_512_fp32.pth',
    'network-bsds500.pth',
    'upernet_global_small.pth',
]
for name in names:
    command = f'wget https://huggingface.co/lllyasviel/ControlNet/resolve/main/annotator/ckpts/{name} -O {name}'
    out_path = pathlib.Path(f'ControlNet/annotator/ckpts/{name}')
    if out_path.exists():
        continue
    subprocess.run(shlex.split(command), cwd='ControlNet/annotator/ckpts/')

from gradio_canny2image import create_demo as create_demo_canny
from gradio_depth2image import create_demo as create_demo_depth
from gradio_fake_scribble2image import create_demo as create_demo_fake_scribble
from gradio_hed2image import create_demo as create_demo_hed
from gradio_hough2image import create_demo as create_demo_hough
from gradio_normal2image import create_demo as create_demo_normal
from gradio_pose2image import create_demo as create_demo_pose
from gradio_scribble2image import create_demo as create_demo_scribble
from gradio_scribble2image_interactive import \
    create_demo as create_demo_scribble_interactive
from gradio_seg2image import create_demo as create_demo_seg
from model import (DEFAULT_BASE_MODEL_FILENAME, DEFAULT_BASE_MODEL_REPO,
                   DEFAULT_BASE_MODEL_URL, Model)

MAX_IMAGES = 4
ALLOW_CHANGING_BASE_MODEL = 'hysts/ControlNet-with-other-models'

model = Model()

with gr.Blocks(css='style.css') as demo:
    with gr.Tabs():
        with gr.TabItem('Canny'):
            create_demo_canny(model.process_canny, max_images=MAX_IMAGES)
        with gr.TabItem('Hough'):
            create_demo_hough(model.process_hough, max_images=MAX_IMAGES)
        with gr.TabItem('HED'):
            create_demo_hed(model.process_hed, max_images=MAX_IMAGES)
        with gr.TabItem('Scribble'):
            create_demo_scribble(model.process_scribble, max_images=MAX_IMAGES)
        with gr.TabItem('Scribble Interactive'):
            create_demo_scribble_interactive(
                model.process_scribble_interactive, max_images=MAX_IMAGES)
        with gr.TabItem('Fake Scribble'):
            create_demo_fake_scribble(model.process_fake_scribble,
                                      max_images=MAX_IMAGES)
        with gr.TabItem('Pose'):
            create_demo_pose(model.process_pose, max_images=MAX_IMAGES)
        with gr.TabItem('Segmentation'):
            create_demo_seg(model.process_seg, max_images=MAX_IMAGES)
        with gr.TabItem('Depth'):
            create_demo_depth(model.process_depth, max_images=MAX_IMAGES)
        with gr.TabItem('Normal map'):
            create_demo_normal(model.process_normal, max_images=MAX_IMAGES)

    import gradio as gr
import torch

DEFAULT_BASE_MODEL_URL = 'openai/clip-vit-base-patch32'
DEFAULT_BASE_MODEL_REPO = 'openai/clip'
DEFAULT_BASE_MODEL_FILENAME = 'clip_vit_base_patch32.pt'
ALLOW_CHANGING_BASE_MODEL = True

class Model:
    def __init__(self, base_model_url):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = torch.load(base_model_url, map_location=self.device).to(self.device)

    def generate_images(self, text, num_images=1, clip_args={}):
        # Function to generate images using the base model
        pass
    
    def set_base_model(self, base_model_repo, base_model_filename):
        # Function to set the base model
        pass

model = Model(DEFAULT_BASE_MODEL_URL)

import gradio as gr
import torch

DEFAULT_BASE_MODEL_URL = 'openai/clip-vit-base-patch32'
DEFAULT_BASE_MODEL_REPO = 'openai/clip'
DEFAULT_BASE_MODEL_FILENAME = 'clip_vit_base_patch32.pt'
ALLOW_CHANGING_BASE_MODEL = True

class Model:
    def __init__(self, base_model_url):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = torch.load(base_model_url, map_location=self.device).to(self.device)

    def generate_images(self, text, num_images=1, clip_args={}):
        # Function to generate images using the base model
        pass
    
    def set_base_model(self, base_model_repo, base_model_filename):
        # Function to set the base model
        pass

model = Model(DEFAULT_BASE_MODEL_URL)

with gr.Accordion(label='Base model', open=False):
    current_base_model = gr.Textbox(label='Current base model', value=DEFAULT_BASE_MODEL_URL)
    base_models = [
        {"label": "ControlNet v1.5", "value": ["openai-diffusion/ControlNet", "cldm_v15.pt"]},
        {"label": "ControlNet v2.0", "value": ["openai-diffusion/ControlNet", "cldm_v20.pt"]},
        {"label": "CLIPDraw", "value": ["openai-diffusion/CLIPDraw", "clipdraw.pt"]},
        {"label": "StyleGAN2-ADA", "value": ["NVLabs/stylegan2-ada", "ffhq.pkl"]},
    ]
    base_model_dropdown = gr.Dropdown(label='Select a base model', choices=base_models, default=base_models[0]['value'])
    change_base_model_button = gr.Button('Change base model')
    gr.Description('''- You can use other base models by selecting from the dropdown menu.
The base model must be compatible with Stable Diffusion v1.5.''')

change_base_model_button.onclick(lambda: model.set_base_model(base_model_dropdown.value[0], base_model_dropdown.value[1]))
demo = gr.Interface(fn=model.generate_images, inputs=["text"], outputs="image", capture_session=True)

gr.Interface(lambda: None, [current_base_model, base_model_dropdown, change_base_model_button], title='Change base model').launch(share=True) 
