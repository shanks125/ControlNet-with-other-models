Convert the below code so instead of user inputting the value for DEFAULT_BASE_MODEL_REPO and DEFAULT_BASE_MODEL_URL, there is a single drop-down menu with multiple options that user can select based on their preferences.

with gr.Accordion(label='Base model', open=False):
        current_base_model = gr.Text(label='Current base model',
                                     value=DEFAULT_BASE_MODEL_URL)
        with gr.Row():
            base_model_repo = gr.Text(label='Base model repo',
                                      max_lines=1,
                                      placeholder=DEFAULT_BASE_MODEL_REPO,
                                      interactive=ALLOW_CHANGING_BASE_MODEL)
            base_model_filename = gr.Text(
                label='Base model file',
                max_lines=1,
                placeholder=DEFAULT_BASE_MODEL_FILENAME,
                interactive=ALLOW_CHANGING_BASE_MODEL)
        change_base_model_button = gr.Button('Change base model')
        gr.Markdown(
            '''- You can use other base models by specifying the repository name and filename.
The base model must be compatible with Stable Diffusion v1.5.''')

    change_base_model_button.click(fn=model.set_base_model,
                                   inputs=[
                                       base_model_repo,
                                       base_model_filename,
                                   ],
                                   outputs=current_base_model)

demo.queue(api_open=False).launch()
