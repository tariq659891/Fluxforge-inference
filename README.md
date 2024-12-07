# FluxForge 🚀

FluxForge is a memory-efficient implementation of Flux/Schnell models with support for quantization, IP-Adapter, LoRA, and ControlNet (coming soon). Run high-quality image generation models with less than 17GB VRAM!

## 🌟 Features

- ✨ Memory-efficient inference (< 17GB VRAM) - Diffuser based IP adapter implementation. 
- 🔧 Multiple model loading options (base, quantized, safetensors)
- 📊 8-bit quantization support
- 🎯 IP-Adapter support (experimental)
- 🔄 LoRA and ControlNet support (coming soon)
- 🚂 Training script (coming soon)

## 🛠️ Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

## 🚀 Usage

### Basic Configuration

## Model Setup
Before running the inference code, you need to convert the model to FP8 format for optimal memory usage:

1. First, run the conversion script:
```python
python convert.py --model_path path/to/flux_model --output_path path/to/flux_model/flux-fp8 --quantization_type qfloat8
```

2. After conversion, update your config to use the quantized model in the fluxforge_main.py or externally:

```python
from fluxforge_main import EnhancedFluxForge, ModelConfig, ModelType

# Basic configuration
config = ModelConfig(
    model_type=ModelType.FLUX,
    model_path="path/to/flux_model",
    enable_quantization=True,
    transformer_path="path/to/flux_model/flux-fp8",
    transformer_loading_mode="quantized"
)

# Initialize and run
forge = FluxForge(config)
image = forge.generate(
    prompt="A majestic statue clutching a glowing 'FluxForge' sign, standing proudly at the heart of a bustling railway station, its presence commanding attention amidst the flow of travelers.",
    width=1024,
    height=1024
)
image.save('output.png')
```
![output1](https://github.com/user-attachments/assets/f05f0283-343d-4cad-8701-5a556e096938)

### Configuration Options

The ModelConfig class supports various loading strategies:

```python
# 1. You can Load quantized safetensors and base model with quantization but prefered mothod is pre-quantized model from convert.py

config = ModelConfig(
    model_path="path/to/flux_model",
    transformer_path="path/to/flux_model/flux-fp8",
    transformer_loading_mode="quantized"
)
```

### IP-Adapter Configuration

```python
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
    seed=123456789,
    image=image
)

output.save('luck5.png')
forge.memory_tracker.print_memory_stats("After Image Generation")
```
The IP adapter is still not functioning as expected. I'm working on aligning its behavior to match the results of the IP adapter in ComfyUI for similar outcomes.
![statue](https://github.com/user-attachments/assets/d355afbb-ab50-40ac-be8f-e3d637cbefac) 
![luck3](https://github.com/user-attachments/assets/c1fb6bfe-105e-45ae-bf41-43b24b4f3dfb)
![luck5](https://github.com/user-attachments/assets/43a527c9-c0a2-49e8-9b2c-b05ddc336c28)

### Configuration Parameters

| Parameter | Description | Options |
|-----------|-------------|----------|
| `model_type` | Model type | `ModelType.FLUX`, `ModelType.FLUX_IP` |
| `model_path` | Path to base model | Path string |
| `transformer_path` | Path to transformer | Path string (optional) |
| `dtype` | Model dtype | `torch.bfloat16` (default) |
| `enable_quantization` | Enable quantization | `True`/`False` |
| `quantization_type` | Type of quantization | `"qfloat8"`, `"qint8"` |
| `transformer_loading_mode` | Loading strategy | `"base"`, `"quantized"`, `"safetensors"` |
| `use_freeze` | Freeze model weights | `True`/`False` |
| `device_map` | Device mapping | `"auto"` (default) |

### Loading Mode Combinations

- **Base Loading**: Set `transformer_loading_mode="base"` and `enable_quantization=False`
- **Quantization on Load**: Set `transformer_loading_mode="base"` and `enable_quantization=True`
- **Pre-quantized Model**: Set `transformer_loading_mode="quantized"`
- **Safetensors Loading**: Set `transformer_loading_mode="safetensors"` with `.safetensors` file

## 🔜 Coming Soon

- LoRA support
- ControlNet integration
- Sub-16GB VRAM optimization
- Training scripts
- Full IP-Adapter integration in diffusers

## ⚠️ Known Issues

- IP-Adapter is currently experimental
- Training scripts under development
- Some features may require slightly more VRAM than targeted

# Important Notice

## License
This repository uses Flux-dev, a non-commercial model by [Black Forest Labs](https://github.com/black-forest-labs/flux). By using this code, you agree to comply with the Flux-dev license agreement and terms of use. Please review the complete license terms at their [official repository](https://github.com/black-forest-labs/flux).


## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📬 Contact

For questions and support, please open an issue in the GitHub repository.

---
**Note**: This project is under active development. Features and memory requirements may change as we continue to optimize the implementation.
