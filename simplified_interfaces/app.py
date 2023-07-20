# Simplified interface to be used with Gradio client (for simple questions only).

import torch
import gradio as gr
from gradio.themes.utils import colors, fonts, sizes

from conversation import Chat

# videochat
from utils.config import Config
from utils.easydict import EasyDict
from models.videochat import VideoChat


# ========================================
#             Model Initialization
# ========================================
def init_model():
    print("Initializing VideoChat")
    config_file = "configs/config.json"
    cfg = Config.from_file(config_file)
    model = VideoChat(config=cfg.model)
    model = model.to(torch.device(cfg.device))
    model = model.eval()
    chat = Chat(model)
    print("Initialization Finished")
    return chat


# ========================================
#             Gradio Setting
# ========================================
def gradio_reset(chat_state, img_list):
    if chat_state is not None:
        chat_state.messages = []
    if img_list is not None:
        img_list = []
    return (
        None,
        gr.update(value=None, interactive=True),
        gr.update(value=None, interactive=True),
        gr.update(placeholder="Please upload your video first", interactive=False),
        gr.update(value="Upload & Start Chat", interactive=True),
        chat_state,
        img_list,
    )


class OpenGVLab(gr.themes.base.Base):
    def __init__(
        self,
        *,
        primary_hue=colors.blue,
        secondary_hue=colors.sky,
        neutral_hue=colors.gray,
        spacing_size=sizes.spacing_md,
        radius_size=sizes.radius_sm,
        text_size=sizes.text_md,
        font=(
            fonts.GoogleFont("Noto Sans"),
            "ui-sans-serif",
            "sans-serif",
        ),
        font_mono=(
            fonts.GoogleFont("IBM Plex Mono"),
            "ui-monospace",
            "monospace",
        ),
    ):
        super().__init__(
            primary_hue=primary_hue,
            secondary_hue=secondary_hue,
            neutral_hue=neutral_hue,
            spacing_size=spacing_size,
            radius_size=radius_size,
            text_size=text_size,
            font=font,
            font_mono=font_mono,
        )
        super().set(
            body_background_fill="*neutral_50",
        )


gvlabtheme = OpenGVLab(
    primary_hue=colors.blue,
    secondary_hue=colors.sky,
    neutral_hue=colors.gray,
    spacing_size=sizes.spacing_md,
    radius_size=sizes.radius_sm,
    text_size=sizes.text_md,
)

title = ""
description = ""


def handle_describe_image(
    gr_img,
    chat_state,
    user_message,
    num_beams,
    temperature,
):
    chat_state = EasyDict(
        {"system": "", "roles": ("Human", "Assistant"), "messages": [], "sep": "###"}
    )
    img_list = []

    if gr_img is None:
        return None, None, gr.update(interactive=True), chat_state, None

    llm_message, img_list, chat_state = chat.upload_img(gr_img, chat_state, img_list)

    # Step 2: gradio_ask
    if len(user_message) == 0:
        return (
            gr.update(interactive=True, placeholder="Input should not be empty!"),
            chat_state,
        )
    chat_state = chat.ask(user_message, chat_state)

    # Step 3: gradio_answer
    llm_message, llm_message_token, chat_state = chat.answer(
        conv=chat_state,
        img_list=img_list,
        max_new_tokens=1000,
        num_beams=num_beams,
        temperature=temperature,
    )
    llm_message = llm_message.replace("<s>", "")  # handle <s>

    return llm_message, chat_state, img_list


def handle_describe_video(
    gr_video,
    chat_state,
    num_segments,
    user_message,
    num_beams,
    temperature,
):
    chat_state = EasyDict(
        {"system": "", "roles": ("Human", "Assistant"), "messages": [], "sep": "###"}
    )
    img_list = []

    if gr_video is None:
        return None, None, gr.update(interactive=True), chat_state, None

    llm_message, img_list, chat_state = chat.upload_video(
        gr_video, chat_state, img_list, num_segments
    )

    # Step 2: gradio_ask
    if len(user_message) == 0:
        return (
            gr.update(interactive=True, placeholder="Input should not be empty!"),
            chat_state,
        )
    chat_state = chat.ask(user_message, chat_state)

    # Step 3: gradio_answer
    llm_message, llm_message_token, chat_state = chat.answer(
        conv=chat_state,
        img_list=img_list,
        max_new_tokens=1000,
        num_beams=num_beams,
        temperature=temperature,
    )
    llm_message = llm_message.replace("<s>", "")  # handle <s>

    return llm_message, chat_state, img_list


with gr.Blocks(
    title="VideoChat!",
    theme=gvlabtheme,
    css="#chatbot {overflow:auto; height:500px;} #InputVideo {overflow:visible; height:320px;} footer {visibility: none}",
) as demo:
    gr.Markdown(title)
    gr.Markdown(description)

    with gr.Column():
        with gr.Column(visible=True) as video_upload:
            with gr.Column(elem_id="image") as img_part:
                with gr.Row():
                    with gr.Column():
                        up_video = gr.Video(
                            interactive=True, include_audio=True, elem_id="video_upload"
                        )

                        describe_video_button = gr.Button(
                            value="Describe Video", interactive=True, variant="primary"
                        )
                    with gr.Column():
                        up_image = gr.Image(
                            type="pil", interactive=True, elem_id="image_upload"
                        )

                        describe_image_button = gr.Button(
                            value="Describe Image", interactive=True, variant="primary"
                        )

            num_beams = gr.Slider(
                minimum=1,
                maximum=10,
                value=1,
                step=1,
                interactive=True,
                label="beam search numbers",
            )

            temperature = gr.Slider(
                minimum=0.1,
                maximum=2.0,
                value=1.0,
                step=0.1,
                interactive=True,
                label="Temperature",
            )

            num_segments = gr.Slider(
                minimum=8,
                maximum=64,
                value=8,
                step=1,
                interactive=True,
                label="Video Segments",
            )

        with gr.Column(visible=True) as input_raws:
            chat_state = gr.State(
                EasyDict(
                    {
                        "system": "",
                        "roles": ("Human", "Assistant"),
                        "messages": [],
                        "sep": "###",
                    }
                )
            )
            img_list = gr.State()

            with gr.Column():
                text_input = gr.Textbox(
                        "describe the image/video in details",
                        show_label=False,
                    ).style(container=False)
                llama_output = gr.Textbox(
                    lines=5, placeholder="Chat goes here...", readonly=True
                )


    chat = init_model()

    describe_image_button.click(
        handle_describe_image,
        [
            up_image,
            chat_state,
            text_input,
            num_beams,
            temperature,
        ],
        [llama_output, chat_state, img_list],
    )

    describe_video_button.click(
        handle_describe_video,
        [
            up_video,
            chat_state,
            num_segments,
            text_input,
            num_beams,
            temperature,
        ],
        [llama_output, chat_state, img_list],
    )
demo.launch(server_name="0.0.0.0", enable_queue=True)
