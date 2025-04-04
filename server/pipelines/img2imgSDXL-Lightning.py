from diffusers import (
    AutoPipelineForImage2Image,
    AutoencoderTiny,
    AutoencoderKL,
    UNet2DConditionModel,
    EulerDiscreteScheduler,
)
from compel import Compel, ReturnedEmbeddingsType
import torch

try:
    import intel_extension_for_pytorch as ipex  # type: ignore
except:
    pass

from safetensors.torch import load_file
from huggingface_hub import hf_hub_download
from config import Args
from pydantic import BaseModel, Field
from PIL import Image
from util import ParamsModel
import math
from pruna import SmashConfig, smash

base = "stabilityai/stable-diffusion-xl-base-1.0"
repo = "ByteDance/SDXL-Lightning"
ckpt = "sdxl_lightning_2step_unet.safetensors"
taesd_model = "madebyollin/taesdxl"
NUM_STEPS = 2

default_prompt = "close-up photography of old man standing in the rain at night, in a street lit by lamps, leica 35mm summilux"
default_negative_prompt = "blurry, low quality, render, 3D, oversaturated"
page_content = """
<h1 class="text-3xl font-bold">Real-Time SDXL Lightning</h1>
<h3 class="text-xl font-bold">Image-to-Image</h3>
<p class="text-sm">
    This demo showcases
    <a
    href="https://huggingface.co/stabilityai/sdxl-turbo"
    target="_blank"
    class="text-blue-500 underline hover:no-underline">SDXL Turbo</a>
Image to Image pipeline using
    <a
    href="https://huggingface.co/docs/diffusers/main/en/using-diffusers/sdxl_turbo"
    target="_blank"
    class="text-blue-500 underline hover:no-underline">Diffusers</a
    > with a MJPEG stream server.
</p>
<p class="text-sm text-gray-500">
    Change the prompt to generate different images, accepts <a
    href="https://github.com/damian0815/compel/blob/main/doc/syntax.md"
    target="_blank"
    class="text-blue-500 underline hover:no-underline">Compel</a
    > syntax.
</p>
"""


class Pipeline:
    class Info(BaseModel):
        name: str = "img2img"
        title: str = "Image-to-Image SDXL-Lightning"
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
        negative_prompt: str = Field(
            default_negative_prompt,
            title="Negative Prompt",
            field="textarea",
            id="negative_prompt",
            hide=True,
        )
        seed: int = Field(
            2159232, min=0, title="Seed", field="seed", hide=True, id="seed"
        )
        steps: int = Field(
            1, min=1, max=10, title="Steps", field="range", hide=True, id="steps"
        )
        width: int = Field(
            1024, min=2, max=15, title="Width", disabled=True, hide=True, id="width"
        )
        height: int = Field(
            1024, min=2, max=15, title="Height", disabled=True, hide=True, id="height"
        )
        guidance_scale: float = Field(
            0.0,
            min=0,
            max=1,
            step=0.001,
            title="Guidance Scale",
            field="range",
            hide=True,
            id="guidance_scale",
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

    def __init__(self, args: Args, device: torch.device, torch_dtype: torch.dtype):
        if args.taesd:
            vae = AutoencoderTiny.from_pretrained(
                taesd_model, torch_dtype=torch_dtype, use_safetensors=True
            )
        else:
            vae = AutoencoderKL.from_pretrained(
                "madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch_dtype
            )

        unet = UNet2DConditionModel.from_config(base, subfolder="unet")
        unet.load_state_dict(load_file(hf_hub_download(repo, ckpt), device=device.type))
        self.pipe = AutoPipelineForImage2Image.from_pretrained(
            base,
            unet=unet,
            torch_dtype=torch_dtype,
            variant="fp16",
            safety_checker=False,
            vae=vae,
        )
        # Ensure sampler uses "trailing" timesteps.
        self.pipe.scheduler = EulerDiscreteScheduler.from_config(
            self.pipe.scheduler.config, timestep_spacing="trailing"
        )

        if args.pruna:
            # Create and smash your model
            smash_config = SmashConfig()
            smash_config["cacher"] = "deepcache"
            smash_config["compiler"] = "stable_fast"
            self.pipe = smash(model=self.pipe, smash_config=smash_config)

        if args.sfast:
            from sfast.compilers.stable_diffusion_pipeline_compiler import (
                compile,
                CompilationConfig,
            )

            config = CompilationConfig.Default()
            config.enable_xformers = True
            config.enable_triton = True
            config.enable_cuda_graph = True
            self.pipe = compile(self.pipe, config=config)

        self.pipe.set_progress_bar_config(disable=True)
        self.pipe.to(device=device, dtype=torch_dtype)
        if device.type != "mps":
            self.pipe.unet.to(memory_format=torch.channels_last)

        if args.torch_compile:
            print("Running torch compile")
            self.pipe.unet = torch.compile(
                self.pipe.unet, mode="reduce-overhead", fullgraph=True
            )
            self.pipe.vae = torch.compile(
                self.pipe.vae, mode="reduce-overhead", fullgraph=True
            )
            self.pipe(
                prompt="warmup",
                image=[Image.new("RGB", (768, 768))],
            )

        if args.compel:
            self.pipe.compel_proc = Compel(
                tokenizer=[self.pipe.tokenizer, self.pipe.tokenizer_2],
                text_encoder=[self.pipe.text_encoder, self.pipe.text_encoder_2],
                returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
                requires_pooled=[False, True],
            )

    def predict(self, params: "Pipeline.InputParams") -> Image.Image:
        generator = torch.manual_seed(params.seed)
        prompt = params.prompt
        negative_prompt = params.negative_prompt
        prompt_embeds = None
        pooled_prompt_embeds = None
        negative_prompt_embeds = None
        negative_pooled_prompt_embeds = None
        if hasattr(self.pipe, "compel_proc"):
            _prompt_embeds, pooled_prompt_embeds = self.pipe.compel_proc(
                [params.prompt, params.negative_prompt]
            )
            prompt = None
            negative_prompt = None
            prompt_embeds = _prompt_embeds[0:1]
            pooled_prompt_embeds = pooled_prompt_embeds[0:1]
            negative_prompt_embeds = _prompt_embeds[1:2]
            negative_pooled_prompt_embeds = pooled_prompt_embeds[1:2]

        steps = params.steps
        strength = params.strength
        if int(steps * strength) < 1:
            steps = math.ceil(1 / max(0.10, strength))

        results = self.pipe(
            image=params.image,
            prompt=prompt,
            negative_prompt=negative_prompt,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            generator=generator,
            strength=strength,
            num_inference_steps=steps,
            guidance_scale=params.guidance_scale,
            width=params.width,
            height=params.height,
            output_type="pil",
        )

        return results.images[0]
