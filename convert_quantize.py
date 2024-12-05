from optimum.quanto import qfloat8
import torch
from diffusers import FluxTransformer2DModel
from optimum.quanto import QuantizedDiffusersModel
class QuantizedFlux2DModel(QuantizedDiffusersModel):
    base_class = FluxTransformer2DModel

transformer = FluxTransformer2DModel.from_pretrained("flux_model", subfolder="transformer", torch_dtype=torch.bfloat16)
transformer = QuantizedFlux2DModel.quantize(transformer, weights=qfloat8)

transformer.save_pretrained("flux_model/flux-fp8")
