from dataclasses import dataclass
from enum import Enum
import torch
from diffusers import FlowMatchEulerDiscreteScheduler, AutoencoderKL
from diffusers.models.transformers.transformer_flux import FluxTransformer2DModel
from diffusers.pipelines.flux.pipeline_flux import FluxPipeline
from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5TokenizerFast
from optimum.quanto import freeze, qfloat8, qint8, quantize, QuantizedDiffusersModel
from typing import Optional, Literal
import gc
from PIL import Image

@dataclass
class ModelConfig:
    model_path: str = "flux_model"
    transformer_path: Optional[str] = None
    dtype: torch.dtype = torch.bfloat16
    enable_quantization: bool = False
    quantization_type: Literal["qfloat8", "qint8"] = "qfloat8"
    use_freeze: bool = False
    device_map: str = "auto"
    transformer_loading_mode: Literal["base", "quantized"] = "base"

class QuantizedFluxTransformer2DModel(QuantizedDiffusersModel):
    base_class = FluxTransformer2DModel

class EnhancedFluxForge:
    def __init__(self, config: ModelConfig):
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.pipe = None
        self._setup_pipeline()

    def _cleanup(self):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    def _encode_prompt(self, prompt: str, num_images_per_prompt: int = 1):
        # Load CLIP
        text_encoder = CLIPTextModel.from_pretrained(
            self.config.model_path,
            subfolder="text_encoder",
            torch_dtype=self.config.dtype,
            device_map=self.config.device_map
        )
        tokenizer = CLIPTokenizer.from_pretrained(
            self.config.model_path,
            subfolder="tokenizer"
        )

        # Process with CLIP
        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            pooled_prompt_embeds = text_encoder(
                text_inputs.input_ids,
                output_hidden_states=False
            ).pooler_output

        # Load T5
        text_encoder_2 = T5EncoderModel.from_pretrained(
            self.config.model_path,
            subfolder="text_encoder_2",
            torch_dtype=self.config.dtype,
            device_map=self.config.device_map
        )
        tokenizer_2 = T5TokenizerFast.from_pretrained(
            self.config.model_path,
            subfolder="tokenizer_2"
        )

        # Process with T5
        text_inputs_2 = tokenizer_2(
            prompt,
            padding="max_length",
            max_length=512,
            truncation=True,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            prompt_embeds = text_encoder_2(
                text_inputs_2.input_ids,
                output_hidden_states=False
            )[0]

            # Handle batch size
            batch_size = 1
            prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
            prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, -1, prompt_embeds.shape[-1])
            pooled_prompt_embeds = pooled_prompt_embeds.repeat(1, num_images_per_prompt, 1)
            pooled_prompt_embeds = pooled_prompt_embeds.view(batch_size * num_images_per_prompt, -1)

        # Cleanup
        del text_encoder, text_encoder_2, tokenizer, tokenizer_2
        self._cleanup()

        return prompt_embeds, pooled_prompt_embeds

    def _load_transformer(self):
        if self.config.transformer_loading_mode == "quantized":
            transformer = QuantizedFluxTransformer2DModel.from_pretrained(
                self.config.transformer_path
            ).to(dtype=self.config.dtype)
        else:
            model_path = self.config.transformer_path or self.config.model_path
            transformer = FluxTransformer2DModel.from_pretrained(
                model_path,
                subfolder="transformer"
            ).to(dtype=self.config.dtype)

            if self.config.enable_quantization:
                quant_type = qfloat8 if self.config.quantization_type == "qfloat8" else qint8
                quantize(transformer, weights=quant_type)

        if self.config.use_freeze:
            freeze(transformer)

        return transformer.to(self.device)

    def _setup_pipeline(self):
        transformer = self._load_transformer()

        scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            self.config.model_path,
            subfolder="scheduler",
            torch_dtype=self.config.dtype
        )

        vae = AutoencoderKL.from_pretrained(
            self.config.model_path,
            subfolder="vae",
            torch_dtype=self.config.dtype
        )

        self.pipe = FluxPipeline(
            scheduler=scheduler,
            text_encoder=None,
            tokenizer=None,
            text_encoder_2=None,
            tokenizer_2=None,
            vae=vae,
            transformer=transformer
        ).to(self.device)

    def generate(
            self,
            prompt: str,
            width: int = 1024,
            height: int = 1024,
            num_inference_steps: int = 20,
            guidance_scale: float = 7.5,
            num_images_per_prompt: int = 1,
            seed: Optional[int] = None
    ) -> Image.Image:
        """
        Generate an image based on the given prompt.

        Args:
            prompt (str): The text prompt to generate the image from
            width (int): Width of the generated image
            height (int): Height of the generated image
            num_inference_steps (int): Number of denoising steps
            guidance_scale (float): Scale for classifier-free guidance
            num_images_per_prompt (int): Number of images to generate
            seed (Optional[int]): Random seed for generation

        Returns:
            PIL.Image.Image: The generated image
        """
        # Set seed if provided
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)
        else:
            generator = None

        # Get embeddings
        prompt_embeds, pooled_prompt_embeds = self._encode_prompt(
            prompt,
            num_images_per_prompt=num_images_per_prompt
        )

        # Generate image
        output = self.pipe(
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            width=width,
            height=height,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            generator=generator
        )

        return output.images[0]


if __name__ == "__main__":
    # Basic configuration
    config = ModelConfig(
        model_path="flux_model",
        enable_quantization=True,
        transformer_path="flux_model/flux-fp8",
        transformer_loading_mode="quantized"
    )

    # Initialize and generate
    forge = EnhancedFluxForge(config)
    image = forge.generate(
        prompt="A majestic statue clutching a glowing 'FluxForge' sign, standing proudly at the heart of a bustling railway station, its presence commanding attention amidst the flow of travelers.",
        width=1024,
        height=1024,
        seed=42  # optional
    )
    image.save('output.png')
