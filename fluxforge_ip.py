from dataclasses import dataclass
from enum import Enum
import torch
import torch.nn as nn
import diffusers
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
from ip_adapter.attention_flux import FluxIPAttnProcessor2_0_fluxforge3

import gc
from optimum.quanto import freeze, qfloat8, quantize, QTensor, qint4

def flush():
    torch.cuda.empty_cache()
gc.collect()

class ModelType(Enum):
    FLUX = "flux"
    FLUX_IP = "flux_ip"

def patch_transformer_forward():
    """Monkey patch the FluxTransformerBlock forward method"""
    # Store original forward method in case we need to restore
    original_forward = diffusers.models.transformers.transformer_flux.FluxTransformerBlock.forward

    def new_forward(
            self,
            hidden_states: torch.FloatTensor,
            encoder_hidden_states: torch.FloatTensor,
            temb: torch.FloatTensor,
            image_rotary_emb=None,
            joint_attention_kwargs=None,
    ):
        # Get modulation parameters
        norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(hidden_states, emb=temb)
        norm_encoder_hidden_states, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = self.norm1_context(
            encoder_hidden_states, emb=temb
        )

        joint_attention_kwargs = joint_attention_kwargs or {}

        # Get attention outputs
        outputs = self.attn(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=norm_encoder_hidden_states,
            image_rotary_emb=image_rotary_emb,
            temb=temb,
            **joint_attention_kwargs,
        )

        if isinstance(outputs, tuple) and len(outputs) == 3:
            attn_output, context_attn_output, ip_attention = outputs
        else:
            attn_output, context_attn_output = outputs
            ip_attention = None

        # Process base attention with gates first
        attn_output = gate_msa.unsqueeze(1) * attn_output
        hidden_states = hidden_states + attn_output #img = img + img_mod1.gate * attn.img_attn.proj(img_attn)

        # Process MLP path
        norm_hidden_states = self.norm2(hidden_states)
        norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
        ff_output = self.ff(norm_hidden_states)
        ff_output = gate_mlp.unsqueeze(1) * ff_output
        hidden_states = hidden_states + ff_output

        # Process encoder states
        context_attn_output = c_gate_msa.unsqueeze(1) * context_attn_output
        encoder_hidden_states = encoder_hidden_states + context_attn_output

        norm_encoder_hidden_states = self.norm2_context(encoder_hidden_states)
        norm_encoder_hidden_states = norm_encoder_hidden_states * (1 + c_scale_mlp[:, None]) + c_shift_mlp[:, None]
        context_ff_output = self.ff_context(norm_encoder_hidden_states)
        encoder_hidden_states = encoder_hidden_states + c_gate_mlp.unsqueeze(1) * context_ff_output

        # Add IP attention last, after all processing
        if ip_attention is not None and 'ip_scale' in joint_attention_kwargs:
            ip_scale = joint_attention_kwargs['ip_scale']
            hidden_states = hidden_states + ip_scale * ip_attention

        if encoder_hidden_states.dtype == torch.float16:
            encoder_hidden_states = encoder_hidden_states.clip(-65504, 65504)

        return encoder_hidden_states, hidden_states

    # Apply the patch only to FluxTransformerBlock
    diffusers.models.transformers.transformer_flux.FluxTransformerBlock.forward = new_forward

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
    model_type: ModelType = ModelType.FLUX_IP
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

    def get_image_proj(self, pil_image: Union[Image.Image, List[Image.Image]]):
        # Load CLIP models if not loaded
        self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            "openai/clip-vit-large-patch14"
        ).to(self.device, dtype=torch.float16)
        self.clip_image_processor = CLIPImageProcessor()

        with safe_open(self.config.ip_ckpt, framework="pt", device="cpu") as f:
            checkpoint = {key: f.get_tensor(key) for key in f.keys()}

        # Setup image projection model
        self.improj = ImageProjModel(4096, 768, 4)
        proj_state_dict = {
            k.replace("ip_adapter_proj_model.", ""): v
            for k, v in checkpoint.items()
            if k.startswith("ip_adapter_proj_model")
        }
        self.improj.load_state_dict(proj_state_dict)
        self.improj = self.improj.to(self.device, dtype=torch.bfloat16)

        # Handle single image input
        if isinstance(pil_image, Image.Image):
            pil_image = [pil_image]

        # Get image embeddings
        clip_image = self.clip_image_processor(
            images=pil_image,
            return_tensors="pt"
        ).pixel_values

        clip_image = clip_image.to(self.image_encoder.device)
        clip_image_embeds = self.image_encoder(
            clip_image
        ).image_embeds.to(
            device=self.device,
            dtype=torch.bfloat16,
        )

        # Project through the image projection model
        image_proj = self.improj(clip_image_embeds)
        return image_proj



    def get_image_embeddings_with_cleanup(self, pil_image: Union[Image.Image, List[Image.Image]], content_prompt_embeds=None):
        """Separate method to get image embeddings and clean up the models"""
        from src.flux.util import load_checkpoint
        try:
            print("\nLoading image encoder and processor...")
            # Load image encoder temporarily
            self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(
                self.config.image_encoder_path
            ).to(self.device, dtype=torch.float16)
            self.clip_image_processor = CLIPImageProcessor()

            # Load projection model temporarily
            checkpoint = load_checkpoint(local_path=self.config.ip_ckpt, repo_id=None, name=None)

            self.image_proj_model = ImageProjModel(4096, 768, 4)

            prefix = "double_blocks."
            blocks = {}
            proj = {}
            for key, value in checkpoint.items():
                if key.startswith(prefix):
                    blocks[key[len(prefix):].replace('.processor.', '.')] = value
                if key.startswith("ip_adapter_proj_model"):
                    proj[key[len("ip_adapter_proj_model."):]] = value

            self.image_proj_model.load_state_dict(proj)
            self.image_proj_model = self.image_proj_model.to(self.device, dtype=torch.bfloat16)

            # Get embeddings
            if isinstance(pil_image, Image.Image):
                pil_image = [pil_image]

            clip_image = self.clip_image_processor(images=pil_image, return_tensors="pt").pixel_values
            clip_image_embeds = self.image_encoder(clip_image.to(self.device)).image_embeds.to(
                device=self.device, dtype=self.config.dtype,
            )

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
        patch_transformer_forward()

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
                    processor = FluxIPAttnProcessor2_0_fluxforge3(
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
            transformer=transformer).to(self.device)

        # self.memory_tracker.print_memory_stats("After Pipeline Setup")
        return pipe

    def generate(
            self,
            prompt: str,
            width: int = 1024,
            height: int = 1024,
            guidance_scale: float = 3.5,
            num_inference_steps: int = 100,
            seed: Optional[int] = None,
            image: Optional[Image.Image] = None,
    ) -> Image.Image:
        """
        Generate an image using the Flux model with optional IP adapter support.
        """
        generator = torch.Generator("cuda").manual_seed(seed) if seed is not None else None

        # Get prompt embeddings
        prompt_embeds, pooled_prompt_embeds, text_ids = self.encode_prompt(prompt)

        # Handle IP adapter if image is provided
        ip_checkpoint = None
        image_embeds = None
        if image is not None and self.config.model_type == ModelType.FLUX_IP:
            image_embeds, _, ip_checkpoint = self.get_image_embeddings_with_cleanup(image)

        # Load transformer
        transformer = self.load_transformer(ip_checkpoint)

        # Setup pipeline
        pipe = self.setup_pipeline(transformer)

        # Setup joint attention kwargs for IP adapter
        joint_attention_kwargs = {
            "ip_hidden_states": image_embeds,
            "ip_scale": 1.0
        } if image_embeds is not None else None

        # Generate image
        output = pipe(
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            width=width,
            height=height,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            generator=generator,
            joint_attention_kwargs=joint_attention_kwargs
        ).images[0]

        # Cleanup
        self._aggressive_cleanup()

        return output

def main():
    # Configure for IP-Adapter
    config = ModelConfig(
        model_type=ModelType.FLUX_IP,
        model_path="flux_model",
        enable_quantization=True,
        transformer_path="flux_model/flux-fp8",
        transformer_loading_mode="quantized",
        image_encoder_path="openai/clip-vit-large-patch14",
        ip_ckpt="flux_model/flux_ip_adapter/ip_adapter.safetensors"
    )

    # Initialize forge
    forge = EnhancedFluxForge(config)

    # Load reference image for IP-Adapter
    image = Image.open("assets/example_images/statue.jpg")

    # Generate with IP-Adapter
    output = forge.generate(
        prompt="wearing glasses",
        width=1024,
        height=1024,
        guidance_scale=4.0,
        num_inference_steps=25,
        seed=123485,
        image=image
    )

    output.save('luck5.png')
    forge.memory_tracker.print_memory_stats("After Image Generation")

if __name__ == "__main__":
    main()
