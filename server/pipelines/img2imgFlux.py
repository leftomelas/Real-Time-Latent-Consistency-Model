import torch

from optimum.quanto import freeze, qfloat8, quantize
from transformers.modeling_utils import PreTrainedModel

from diffusers import (
    FlowMatchEulerDiscreteScheduler,
    AutoencoderKL,
    AutoencoderTiny,
    FluxImg2ImgPipeline,
    FluxPipeline,
)

from diffusers import (
    FluxImg2ImgPipeline,
    FluxPipeline,
    FluxTransformer2DModel,
    GGUFQuantizationConfig,
)

try:
    import intel_extension_for_pytorch as ipex  # type: ignore
except:
    pass

import psutil
from config import Args
from pydantic import BaseModel, Field
from PIL import Image
from pathlib import Path
from util import ParamsModel
import math
import gc


# model_path = "black-forest-labs/FLUX.1-dev"
model_path = "black-forest-labs/FLUX.1-schnell"
base_model_path = "black-forest-labs/FLUX.1-schnell"
taesd_path = "madebyollin/taef1"
subfolder = "transformer"
transformer_path = model_path
models_path = Path("models")

default_prompt = "close-up photography of old man standing in the rain at night, in a street lit by lamps, leica 35mm summilux"
default_negative_prompt = "blurry, low quality, render, 3D, oversaturated"
page_content = """
<h1 class="text-3xl font-bold">Real-Time FLUX</h1>

"""


def flush():
    torch.cuda.empty_cache()
    gc.collect()


class Pipeline:
    class Info(BaseModel):
        name: str = "img2img"
        title: str = "Image-to-Image SDXL"
        description: str = "Generates an image from a text prompt"
        input_mode: str = "image"
        page_content: str = page_content

    class InputParams(ParamsModel):
        prompt: str = Field(
            default_prompt,
            title="Prompt",
            field="textarea",
            id="prompt",
        )
        seed: int = Field(
            2159232, min=0, title="Seed", field="seed", hide=True, id="seed"
        )
        steps: int = Field(
            1, min=1, max=15, title="Steps", field="range", hide=True, id="steps"
        )
        width: int = Field(
            256, min=2, max=15, title="Width", disabled=True, hide=True, id="width"
        )
        height: int = Field(
            256, min=2, max=15, title="Height", disabled=True, hide=True, id="height"
        )
        strength: float = Field(
            0.5,
            min=0.25,
            max=1.0,
            step=0.001,
            title="Strength",
            field="range",
            hide=True,
            id="strength",
        )
        guidance: float = Field(
            3.5,
            min=0,
            max=20,
            step=0.001,
            title="Guidance",
            hide=True,
            field="range",
            id="guidance",
        )

    def __init__(self, args: Args, device: torch.device, torch_dtype: torch.dtype):
        # ckpt_path = (
        #     "https://huggingface.co/city96/FLUX.1-dev-gguf/blob/main/flux1-dev-Q2_K.gguf"
        # )
        print("Loading model")
        # ckpt_path: str = "https://huggingface.co/city96/FLUX.1-schnell-gguf/blob/main/flux1-schnell-Q6_K.gguf"
        ckpt_path: str = "https://huggingface.co/city96/FLUX.1-schnell-gguf/blob/main/flux1-schnell-Q4_K_S.gguf"
        transformer = FluxTransformer2DModel.from_single_file(
            ckpt_path,
            quantization_config=GGUFQuantizationConfig(compute_dtype=torch.bfloat16),
            torch_dtype=torch.bfloat16,
        )

        # else:
        pipe = FluxImg2ImgPipeline.from_pretrained(
            # "black-forest-labs/FLUX.1-dev",
            "black-forest-labs/FLUX.1-Schnell",
            transformer=transformer,
            torch_dtype=torch.bfloat16,
        )
        if args.taesd:
            pipe.vae = AutoencoderTiny.from_pretrained(
                taesd_path, torch_dtype=torch.bfloat16, use_safetensors=True
            )
        # pipe.enable_model_cpu_offload()
        pipe = pipe.to(device)

        # pipe.enable_model_cpu_offload()

        self.pipe = pipe
        self.pipe.set_progress_bar_config(disable=True)

        #     vae = AutoencoderKL.from_pretrained(
        #         base_model_path, subfolder="vae", torch_dtype=torch_dtype
        # )

    def predict(self, params: "Pipeline.InputParams") -> Image.Image:
        generator = torch.manual_seed(params.seed)
        steps = params.steps
        strength = params.strength
        prompt = params.prompt
        guidance = params.guidance

        results = self.pipe(
            image=params.image,
            prompt=prompt,
            generator=generator,
            strength=strength,
            num_inference_steps=steps,
            guidance_scale=guidance,
            width=params.width,
            height=params.height,
        )
        return results.images[0]
