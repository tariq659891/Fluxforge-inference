import argparse
from optimum.quanto import qfloat8
import torch
from diffusers import FluxTransformer2DModel
from optimum.quanto import QuantizedDiffusersModel

class QuantizedFlux2DModel(QuantizedDiffusersModel):
    base_class = FluxTransformer2DModel

def main():
    parser = argparse.ArgumentParser(description='Quantize FluxTransformer2DModel to lower precision')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the original model')
    parser.add_argument('--output_path', type=str, required=True, help='Path to save the quantized model')
    parser.add_argument('--quantization_type', type=str, default='qfloat8', choices=['qfloat8'],
                        help='Quantization type to use (currently only qfloat8 supported)')
    
    args = parser.parse_args()
    
    # Load the model
    transformer = FluxTransformer2DModel.from_pretrained(
        args.model_path,
        subfolder="transformer",
        torch_dtype=torch.bfloat16
    )
    
    # Quantize the model
    quantization_type = qfloat8 if args.quantization_type == 'qfloat8' else None
    transformer = QuantizedFlux2DModel.quantize(transformer, weights=quantization_type)
    
    # Save the quantized model
    transformer.save_pretrained(args.output_path)

if __name__ == "__main__":
    main()
