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

    with gr.Accordion(label='Base model', open=False):
    current_base_model = gr.Textbox(label='Current base model', value=DEFAULT_BASE_MODEL_URL)
    
    base_model_repo_options = {
        'Default': 'DEFAULT_BASE_MODEL_REPO_PLACEHOLDER',
        'Anything-v4.0': 'andite/anything-v4.0',
        'Something-v2.0': 'user/something-v2.0',
        'Another-model': 'user/another-model'
    }
    base_model_repo = gr.inputs.Dropdown(label='Base model repo', choices=base_model_repo_options)
    
    base_model_filename_options = {
        'Default': 'DEFAULT_BASE_MODEL_FILENAME_PLACEHOLDER',
        'Filename-1': 'filename1.safetensors',
        'Filename-2': 'filename2.safetensors',
        'Filename-3': 'filename3.safetensors'
    }
    base_model_filename = gr.inputs.Dropdown(label='Base model file', choices=base_model_filename_options)

    change_base_model_button = gr.Button('Change base model')
    gr.Markdown('''- You can use other base models by selecting from the dropdowns.
The base model must be compatible with Stable Diffusion v1.5.''')

    def update_base_model():
        repo = base_model_repo_options[base_model_repo.value]
        filename = base_model_filename_options[base_model_filename.value]
        model.set_base_model(repo, filename)
        current_base_model.update(value=model.base_model_url)

    change_base_model_button.click(update_base_model)
    
demo = gr.Interface(fn=my_function, inputs=[], outputs=[], title="My Interface")
demo.queue(api_open=False).launch()
