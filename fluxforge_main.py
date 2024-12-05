from dataclasses import dataclass
from enum import Enum
import torch
import torch.nn as nn
from diffusers import FlowMatchEulerDiscreteScheduler, AutoencoderKL
from diffusers.models.transformers.transformer_flux import FluxTransformer2DModel
from diffusers.pipelines.flux.pipeline_flux import FluxPipeline
from transformers import (
    CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5TokenizerFast,
    CLIPImageProcessor, CLIPVisionModelWithProjection
)
from optimum.quanto import freeze, qfloat8, qint8, quantize, QuantizedDiffusersModel
from typing import Optional, Union, Dict, Any, Literal, List
import gc
import psutil
from PIL import Image
from safetensors import safe_open
from ip_adapter.attention_flux import FluxIPAttnProcessor2_0_fluxforge2

import gc
from optimum.quanto import freeze, qfloat8, quantize, QTensor, qint4

def flush():
    torch.cuda.empty_cache()
gc.collect()

class ModelType(Enum):
    FLUX = "flux"
    FLUX_IP = "flux_ip"

def patch_attention_forward():
    """Monkey patches the Attention forward method to handle IP-adapter kwargs."""
    from diffusers.models.attention_processor import Attention
    import inspect

    def new_forward(
            self,
            hidden_states: torch.Tensor,
            encoder_hidden_states: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            **cross_attention_kwargs,
    ) -> torch.Tensor:
        # Filter kwargs based on processor's expected parameters
        try:
            attn_parameters = set(inspect.signature(self.processor.__call__).parameters.keys())
        except ValueError:
            attn_parameters = {"self", "hidden_states", "encoder_hidden_states", "attention_mask",
                               "scale", "ip_hidden_states"}

        # Allowlist for special parameters that shouldn't trigger warnings
        quiet_params = {"ip_adapter_masks"}

        # Filter out unexpected kwargs without warnings
        filtered_kwargs = {
            k: v for k, v in cross_attention_kwargs.items()
            if k in attn_parameters or k in quiet_params
        }

        # Process attention with filtered kwargs
        return self.processor(
            self,
            hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=attention_mask,
            **filtered_kwargs,
        )

    # Apply the patch
    Attention.forward = new_forward

@dataclass
class ModelConfig:
    model_type: ModelType = ModelType.FLUX
    model_path: str = "flux_model"
    transformer_path: Optional[str] = None
    use_safetensors: bool = False
    dtype: torch.dtype = torch.bfloat16
    enable_quantization: bool = False
    quantization_type: Literal["qfloat8", "qint8"] = "qfloat8"
    use_freeze: bool = False
    device_map: str = "auto"
    transformer_loading_mode: Literal["base", "quantized", "safetensors"] = "base"
    # IP Adapter specific configs
    image_encoder_path: Optional[str] = None
    ip_ckpt: Optional[str] = None
    num_tokens: int = 4


class ImageProjModel(nn.Module):
    """Projection Model for IP-Adapter"""

    def __init__(self, cross_attention_dim=1024, clip_embeddings_dim=1024, clip_extra_context_tokens=4):
        super().__init__()
        self.cross_attention_dim = cross_attention_dim
        self.clip_extra_context_tokens = clip_extra_context_tokens
        self.proj = nn.Linear(clip_embeddings_dim, self.clip_extra_context_tokens * cross_attention_dim)
        self.norm = nn.LayerNorm(cross_attention_dim)

    def forward(self, image_embeds):
        embeds = self.proj(image_embeds).reshape(-1, self.clip_extra_context_tokens, self.cross_attention_dim)
        return self.norm(embeds)


class MemoryTracker:
    @staticmethod
    def get_memory_stats():
        stats = {
            'cuda_allocated': torch.cuda.memory_allocated() / 1024 ** 2,
            'cuda_reserved': torch.cuda.memory_reserved() / 1024 ** 2,
            'cuda_max_allocated': torch.cuda.max_memory_allocated() / 1024 ** 2,
            'ram_used': psutil.Process().memory_info().rss / 1024 ** 2,
            'ram_percent': psutil.Process().memory_percent()
        }
        return stats

    @staticmethod
    def print_memory_stats(step_name: str):
        stats = MemoryTracker.get_memory_stats()
        print(f"\nMemory Usage at: {step_name}")
        for k, v in stats.items():
            print(f"{k}: {v:.2f} MB")


class QuantizedFluxTransformer2DModel(QuantizedDiffusersModel):
    base_class = FluxTransformer2DModel


class EnhancedFluxForge:
    def __init__(self, config: ModelConfig):
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.memory_tracker = MemoryTracker()
        self.image_encoder = None
        self.image_proj_model = None
        self.clip_image_processor = None
        self._aggressive_cleanup()
        self.memory_tracker.print_memory_stats("Initialization")

    def _aggressive_cleanup(self):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()

    def _unload_model(self, model, model_name: str):
        if model is not None:
            model = model.cpu()
            del model
            self._aggressive_cleanup()

    def encode_prompt(
            self,
            prompt: str,
            prompt_2: Optional[str] = None,
            max_sequence_length: int = 512,
            num_images_per_prompt: int = 1
    ):
        """Encode prompt with enhanced memory management."""
        try:
            self._aggressive_cleanup()
            batch_size = 1  # Since we're handling single prompt

            # CLIP Processing
            print("\nLoading CLIP encoder...")
            text_encoder = CLIPTextModel.from_pretrained(
                self.config.model_path,
                subfolder="text_encoder",
                torch_dtype=self.config.dtype,
                device_map=self.config.device_map
            )
            tokenizer = CLIPTokenizer.from_pretrained(
                self.config.model_path,
                subfolder="tokenizer",
                clean_up_tokenization_spaces=True
            )

            # Process with CLIP
            with torch.no_grad():
                text_inputs = tokenizer(
                    prompt,
                    padding="max_length",
                    max_length=tokenizer.model_max_length,
                    truncation=True,
                    return_overflowing_tokens=False,
                    return_length=False,
                    return_tensors="pt",
                ).to(self.device)

                # Check for truncation in CLIP
                untruncated_ids = tokenizer(
                    prompt,
                    padding="longest",
                    return_tensors="pt"
                ).input_ids.to(self.device)

                if (untruncated_ids.shape[-1] >= text_inputs.input_ids.shape[-1] and
                        not torch.equal(text_inputs.input_ids, untruncated_ids)):
                    removed_text = tokenizer.batch_decode(
                        untruncated_ids[:, tokenizer.model_max_length - 1: -1]
                    )
                    print(f"Warning: CLIP truncated text: {removed_text}")

                pooled_prompt_embeds = text_encoder(
                    text_inputs.input_ids,
                    output_hidden_states=False
                ).pooler_output.detach().clone()

                # Duplicate CLIP embeddings for each generation
                pooled_prompt_embeds = pooled_prompt_embeds.repeat(1, num_images_per_prompt)
                pooled_prompt_embeds = pooled_prompt_embeds.view(
                    batch_size * num_images_per_prompt, -1
                )

            # T5 Processing
            print("\nLoading T5 encoder...")
            text_encoder_2 = T5EncoderModel.from_pretrained(
                self.config.model_path,
                subfolder="text_encoder_2",
                torch_dtype=self.config.dtype,
                device_map=self.config.device_map
            )
            quantize_enable = True
            if quantize_enable:
                print("Quantizing T5")
                quantize(text_encoder_2, weights=qfloat8)
                freeze(text_encoder_2)
                flush()
            tokenizer_2 = T5TokenizerFast.from_pretrained(
                self.config.model_path,
                subfolder="tokenizer_2",
                clean_up_tokenization_spaces=True
            )

            self.memory_tracker.print_memory_stats("After Loading T5 and CLIP")

            # Process with T5
            with torch.no_grad():
                t5_prompt = prompt_2 if prompt_2 is not None else prompt
                text_inputs_2 = tokenizer_2(
                    t5_prompt,
                    padding="max_length",
                    max_length=max_sequence_length,
                    truncation=True,
                    return_tensors="pt"
                ).to(self.device)

                # Check for truncation in T5
                untruncated_ids_2 = tokenizer_2(
                    t5_prompt,
                    padding="longest",
                    return_tensors="pt"
                ).input_ids.to(self.device)

                if (untruncated_ids_2.shape[-1] >= text_inputs_2.input_ids.shape[-1] and
                        not torch.equal(text_inputs_2.input_ids, untruncated_ids_2)):
                    removed_text = tokenizer_2.batch_decode(
                        untruncated_ids_2[:, max_sequence_length - 1: -1]
                    )
                    print(f"Warning: T5 truncated text: {removed_text}")

                prompt_embeds = text_encoder_2(
                    text_inputs_2.input_ids.to(self.device),
                    output_hidden_states=False
                )[0].detach().clone()

                # Duplicate T5 embeddings for each generation
                bs, seq_len, _ = prompt_embeds.shape
                prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
                prompt_embeds = prompt_embeds.view(
                    batch_size * num_images_per_prompt, seq_len, -1
                )

            # Clear CLIP
            self._unload_model(text_encoder, "CLIP")
            del text_inputs, tokenizer, untruncated_ids
            # self._aggressive_cleanup()

            # Clear T5
            self._unload_model(text_encoder_2, "T5")
            del text_inputs_2, tokenizer_2, untruncated_ids_2
            self._aggressive_cleanup()

            # Create text IDs
            text_ids = torch.zeros(
                prompt_embeds.shape[1],
                3,
                device=self.device,
                dtype=self.config.dtype
            )

            # Ensure embeddings are on the correct device
            prompt_embeds = prompt_embeds.to(self.device)
            pooled_prompt_embeds = pooled_prompt_embeds.to(self.device)

            self.memory_tracker.print_memory_stats("After Encoding Completion")

            return prompt_embeds, pooled_prompt_embeds, text_ids

        except Exception as e:
            print(f"Error during encoding: {e}")
            self._aggressive_cleanup()
            raise

    def get_image_embeddings_with_cleanup(self, pil_image: Union[Image.Image, List[Image.Image]], content_prompt_embeds=None):
        """Separate method to get image embeddings and clean up the models"""
        try:
            print("\nLoading image encoder and processor...")
            # Load image encoder temporarily
            self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(
                self.config.image_encoder_path
            ).to(self.device, dtype=self.config.dtype)
            self.clip_image_processor = CLIPImageProcessor()

            # Load projection model temporarily
            with safe_open(self.config.ip_ckpt, framework="pt", device="cpu") as f:
                checkpoint = {key: f.get_tensor(key) for key in f.keys()}

            self.image_proj_model = ImageProjModel(4096, 768, self.config.num_tokens)
            proj_state_dict = {
                k.replace("ip_adapter_proj_model.", ""): v
                for k, v in checkpoint.items()
                if k.startswith("ip_adapter_proj_model")
            }
            self.image_proj_model.load_state_dict(proj_state_dict)
            self.image_proj_model = self.image_proj_model.to(self.device, dtype=self.config.dtype)

            # Get embeddings
            if isinstance(pil_image, Image.Image):
                pil_image = [pil_image]

            clip_image = self.clip_image_processor(images=pil_image, return_tensors="pt").pixel_values
            clip_image_embeds = self.image_encoder(clip_image.to(self.device, dtype=self.config.dtype)).image_embeds

            if content_prompt_embeds is not None:
                clip_image_embeds = clip_image_embeds - content_prompt_embeds

            image_prompt_embeds = self.image_proj_model(clip_image_embeds)
            uncond_image_prompt_embeds = self.image_proj_model(torch.zeros_like(clip_image_embeds))

            # Clean up
            self._unload_model(self.image_encoder, "Image Encoder")
            self._unload_model(self.image_proj_model, "Image Projection Model")
            self.image_encoder = None
            self.image_proj_model = None
            self.clip_image_processor = None
            self._aggressive_cleanup()

            return image_prompt_embeds, uncond_image_prompt_embeds, checkpoint

        except Exception as e:
            print(f"Error during image encoding: {e}")
            self._aggressive_cleanup()
            raise

    def apply_ip_adapter_to_transformer(self, transformer, checkpoint):
        """Apply IP-Adapter processors to transformer"""
        print("\nApplying IP-Adapter processors...")
        ip_attn_procs = {}

        patch_attention_forward()

        # Process transformer blocks
        for i, block in enumerate(transformer.transformer_blocks):
            if hasattr(block, 'attn'):
                name = f"transformer_blocks.{i}"
                # Extract IP state dict for the block
                prefix = f"double_blocks.{i}.processor."
                ip_state_dict = {
                    k.replace(prefix, ''): v
                    for k, v in checkpoint.items()
                    if k.startswith(prefix)
                }
                if ip_state_dict:
                    processor = FluxIPAttnProcessor2_0_fluxforge2(
                        3072, 4096,
                        scale=1.0,
                        num_tokens=self.config.num_tokens
                    )
                    # Map state dict keys
                    new_state_dict = {}
                    for k, v in ip_state_dict.items():
                        if 'ip_adapter_double_stream_k_proj' in k:
                            new_state_dict['to_k_ip.' + k.split('ip_adapter_double_stream_k_proj.')[-1]] = v
                        elif 'ip_adapter_double_stream_v_proj' in k:
                            new_state_dict['to_v_ip.' + k.split('ip_adapter_double_stream_v_proj.')[-1]] = v
                    processor.load_state_dict(new_state_dict, strict=False)
                    ip_attn_procs[f"{name}.attn"] = processor.to(self.device, dtype=self.config.dtype)

        # Apply processors
        for name, module in transformer.named_modules():
            if name in ip_attn_procs:
                if hasattr(module, 'set_processor'):
                    module.set_processor(ip_attn_procs[name])

        return transformer

    def load_transformer(self, ip_checkpoint=None):
        """Load transformer with optional IP-Adapter support"""
        self._aggressive_cleanup()
        try:
            # Load base transformer
            if self.config.transformer_loading_mode == "quantized" and ".safetensors" not in self.config.transformer_path:
                transformer = QuantizedFluxTransformer2DModel.from_pretrained(self.config.transformer_path).to(dtype=self.config.dtype)
                transformer = transformer.to(device=self.device)
            elif ".safetensors" in self.config.transformer_path:
                transformer = FluxTransformer2DModel.from_config(
                    pretrained_model_name_or_path=f"{self.config.model_path}/flux_dev_quantization_map.json",
                    weights_path=self.config.transformer_path,
                    torch_dtype=self.config.dtype
                )
                transformer.to(self.device)
            else:
                model_path = self.config.transformer_path or self.config.model_path
                transformer = FluxTransformer2DModel.from_pretrained(model_path, subfolder="transformer").to(dtype=self.config.dtype).to(device=self.device)

            # Apply quantization if needed
            if self.config.enable_quantization and self.config.transformer_loading_mode != "quantized":
                print("\nApplying quantization...")
                quant_type = qfloat8 if self.config.quantization_type == "qfloat8" else qint8
                quantize(transformer, weights=quant_type)

            # Apply IP-Adapter if checkpoint provided
            if ip_checkpoint is not None:
                transformer = self.apply_ip_adapter_to_transformer(transformer, ip_checkpoint)

            if self.config.use_freeze:
                print("\nFreezing model...")
                freeze(transformer)

            self.memory_tracker.print_memory_stats("After Loading Transformer")
            return transformer

        except Exception as e:
            print(f"Error loading transformer: {e}")
            self._aggressive_cleanup()
            raise

    @torch.inference_mode()
    def get_image_embeds(self, pil_image: Union[Image.Image, List[Image.Image]], content_prompt_embeds=None):
        """Get image embeddings for IP-Adapter"""
        if isinstance(pil_image, Image.Image):
            pil_image = [pil_image]

        clip_image = self.clip_image_processor(images=pil_image, return_tensors="pt").pixel_values
        clip_image_embeds = self.image_encoder(clip_image.to(self.device, dtype=self.config.dtype)).image_embeds

        if content_prompt_embeds is not None:
            clip_image_embeds = clip_image_embeds - content_prompt_embeds

        image_prompt_embeds = self.image_proj_model(clip_image_embeds)
        uncond_image_prompt_embeds = self.image_proj_model(torch.zeros_like(clip_image_embeds))

        return image_prompt_embeds, uncond_image_prompt_embeds

    def setup_pipeline(self, transformer):
        """Setup pipeline with memory tracking."""
        self.memory_tracker.print_memory_stats("Before Pipeline Setup")

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

        pipe = FluxPipeline(
            scheduler=scheduler,
            text_encoder=None,
            tokenizer=None,
            text_encoder_2=None,
            tokenizer_2=None,
            vae=vae,
            transformer=transformer
        ).to(self.device)

        # self.memory_tracker.print_memory_stats("After Pipeline Setup")
        return pipe


# Example usage:
def main():
    # Configure for IP-Adapter
    config = ModelConfig(
        model_type=ModelType.FLUX,
        model_path="flux_model",
        enable_quantization=True, #if mode is already quantized then no quantization will be applied
        transformer_path="flux_model/flux-fp8", #either select quantized .safetensor or normal transformer
        transformer_loading_mode="quantized", #if base then no quantization, onl quantization will be applied based on enable_quantization
        image_encoder_path="openai/clip-vit-large-patch14",
        ip_ckpt="flux_model/flux_ip_adapter/ip_adapter.safetensors"
    )

    # Initialize forge
    forge = EnhancedFluxForge(config)

    # Step 1: Get prompt embeddings
    print("\n=== Getting Text Embeddings ===")
    prompt = "high quality photo of a dog"
    prompt_embeds, pooled_prompt_embeds, text_ids = forge.encode_prompt(prompt)
    forge.memory_tracker.print_memory_stats("After Text Encoding")

    # Step 2: Get image embeddings if using IP-Adapter
    ip_checkpoint = None
    image_embeds = None
    uncond_image_embeds = None

    if config.model_type == ModelType.FLUX_IP:
        print("\n=== Getting Image Embeddings ===")
        image = Image.open("assets/example_images/statue.jpg")
        image_embeds, uncond_image_embeds, ip_checkpoint = forge.get_image_embeddings_with_cleanup(image)
        forge.memory_tracker.print_memory_stats("After Image Encoding")

    # Step 3: Load transformer with IP-Adapter
    print("\n=== Loading Transformer ===")
    transformer = forge.load_transformer(ip_checkpoint)

    # Step 4: Setup pipeline
    pipe = forge.setup_pipeline(transformer)

    # Generate
    joint_attention_kwargs = {
        "ip_hidden_states": image_embeds,
        "scale": 1.0
    } if image_embeds is not None else None

    image = pipe(
        prompt_embeds=prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
        width=1024,
        height=1024,
        guidance_scale=7.5,
        num_inference_steps=20,
        generator=torch.Generator("cuda").manual_seed(42),
        joint_attention_kwargs=joint_attention_kwargs
    ).images[0]

    image.save('output.png')
    forge.memory_tracker.print_memory_stats("After Image Generation")

if __name__ == "__main__":
    main()