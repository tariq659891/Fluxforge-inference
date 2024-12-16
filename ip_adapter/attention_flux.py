import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from diffusers.models.attention_processor import Attention


class FluxIPAttnProcessor2_0_fluxforge3(nn.Module):
    def __init__(self, hidden_size: int, cross_attention_dim: int = None, scale=1.0, num_tokens=4, device=None, dtype=None):
        super().__init__()
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("Requires PyTorch 2.0 or higher")

        # Keep exact same initialization for weight compatibility
        self.hidden_size = hidden_size
        self.cross_attention_dim = cross_attention_dim or 4096

        if not isinstance(num_tokens, (tuple, list)):
            num_tokens = [num_tokens]
        if not isinstance(scale, list):
            scale = [scale] * len(num_tokens)
        self.scale = scale

        # Keep same parameter names
        self.to_k_ip = nn.Linear(self.cross_attention_dim, hidden_size, bias=True)
        self.to_v_ip = nn.Linear(self.cross_attention_dim, hidden_size, bias=True)

        # Keep same initialization
        nn.init.normal_(self.to_k_ip.weight, std=0.02)
        nn.init.normal_(self.to_v_ip.weight, std=0.02)

    def __call__(
            self,
            attn: Attention,
            hidden_states: torch.FloatTensor,
            encoder_hidden_states: torch.FloatTensor = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            image_rotary_emb: Optional[torch.Tensor] = None,
            ip_hidden_states: Optional[torch.FloatTensor] = None,
            temb: Optional[torch.FloatTensor] = None,
            **kwargs,
    ) -> torch.FloatTensor:
        batch_size, _, _ = hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape

        # Store query projection for IP attention later
        hidden_states_query_proj = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        # Keep all the same reshaping and processing
        hidden_states_query_proj = hidden_states_query_proj.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            hidden_states_query_proj = attn.norm_q(hidden_states_query_proj)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # Handle encoder states exactly as before
        if encoder_hidden_states is not None:
            encoder_hidden_states_query_proj = attn.add_q_proj(encoder_hidden_states)
            encoder_hidden_states_key_proj = attn.add_k_proj(encoder_hidden_states)
            encoder_hidden_states_value_proj = attn.add_v_proj(encoder_hidden_states)

            encoder_shapes = [
                encoder_hidden_states_query_proj.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2),
                encoder_hidden_states_key_proj.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2),
                encoder_hidden_states_value_proj.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            ]
            encoder_hidden_states_query_proj, encoder_hidden_states_key_proj, encoder_hidden_states_value_proj = encoder_shapes

            if attn.norm_added_q is not None:
                encoder_hidden_states_query_proj = attn.norm_added_q(encoder_hidden_states_query_proj)
            if attn.norm_added_k is not None:
                encoder_hidden_states_key_proj = attn.norm_added_k(encoder_hidden_states_key_proj)

            query = torch.cat([encoder_hidden_states_query_proj, hidden_states_query_proj], dim=2)
            key = torch.cat([encoder_hidden_states_key_proj, key], dim=2)
            value = torch.cat([encoder_hidden_states_value_proj, value], dim=2)

            if image_rotary_emb is not None:
                from diffusers.models.embeddings import apply_rotary_emb
                query = apply_rotary_emb(query, image_rotary_emb)
                key = apply_rotary_emb(key, image_rotary_emb)

            # Compute base attention
            attention_output = F.scaled_dot_product_attention(
                query, key, value,
                attn_mask=attention_mask,
                dropout_p=0.0,
                is_causal=False
            )

            attention_output = attention_output.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
            attention_output = attention_output.to(query.dtype)

            # Split for encoder and hidden states
            encoder_attention, hidden_attention = attention_output.split(
                [encoder_hidden_states.shape[1], hidden_states.shape[1]],
                dim=1
            )

            # Project outputs
            hidden_states = attn.to_out[0](hidden_attention)
            hidden_states = attn.to_out[1](hidden_states)
            encoder_hidden_states = attn.to_add_out(encoder_attention)

            # Compute IP attention if needed
            ip_attention = None
            if ip_hidden_states is not None:
                ip_key = self.to_k_ip(ip_hidden_states)
                ip_value = self.to_v_ip(ip_hidden_states)

                ip_key = ip_key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
                ip_value = ip_value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

                ip_attention = F.scaled_dot_product_attention(
                    hidden_states_query_proj, ip_key, ip_value,
                    attn_mask=None,
                    dropout_p=0.0,
                    is_causal=False
                )
                ip_attention = ip_attention.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
                ip_attention = ip_attention.to(hidden_states_query_proj.dtype)

            return (hidden_states, encoder_hidden_states, ip_attention)
        else:
            return hidden_states




class FluxIPAttnProcessor2_0_fluxforge2(nn.Module):
    """Flux Attention processor for IP-Adapter mapped to reference implementation."""

    def __init__(
            self, hidden_size: int, cross_attention_dim: int = None, scale=1.0, num_tokens=4, device=None, dtype=None
    ):
        super().__init__()
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("Requires PyTorch 2.0 or higher")

        self.hidden_size = hidden_size
        self.cross_attention_dim = cross_attention_dim or 4096  # Keep 4096 for your setup

        # Handle scale like reference
        if not isinstance(num_tokens, (tuple, list)):
            num_tokens = [num_tokens]
        if not isinstance(scale, list):
            scale = [scale] * len(num_tokens)
        self.scale = scale

        # Keep your naming but use ModuleList structure
        self.to_k_ip = nn.Linear(self.cross_attention_dim, hidden_size, bias=True)
        self.to_v_ip = nn.Linear(self.cross_attention_dim, hidden_size, bias=True)

        # Initialize with small values instead of zeros
        nn.init.zeros_(self.to_k_ip.weight)
        nn.init.zeros_(self.to_k_ip.bias)

        nn.init.zeros_(self.to_v_ip.weight)
        nn.init.zeros_(self.to_v_ip.bias)

    def __call__(
            self,
            attn: Attention,
            hidden_states: torch.FloatTensor,
            encoder_hidden_states: torch.FloatTensor = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            image_rotary_emb: Optional[torch.Tensor] = None,
            ip_hidden_states: Optional[torch.FloatTensor] = None,
            **kwargs,
    ) -> torch.FloatTensor:
        if ip_hidden_states is None:
            raise ValueError("ip_hidden_states is None")

        batch_size, _, _ = hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape

        #----------------------------------- Image ------------------------------------------

        #hidden_states ---> img_mod1
        #hidden_states ---> img_mod1

        # qkv ---> img_qkv = attn.img_attn.qkv(img_modulated)
        hidden_states_query_proj = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)
        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        #multi head --> img_q, img_k, img_v = rearrange(img_qkv, "B L (K H D) -> K B H L D", K=3, H=attn.num_heads, D=attn.head_dim)
        hidden_states_query_proj = hidden_states_query_proj.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # Apply normalizations --> img_q, img_k = attn.img_attn.norm(img_q, img_k, img_v)
        if attn.norm_q is not None:
            hidden_states_query_proj = attn.norm_q(hidden_states_query_proj)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        #----------------------------------- Text ------------------------------------------
        if encoder_hidden_states is not None:
            # qkv
            encoder_hidden_states_query_proj = attn.add_q_proj(encoder_hidden_states)
            encoder_hidden_states_key_proj = attn.add_k_proj(encoder_hidden_states)
            encoder_hidden_states_value_proj = attn.add_v_proj(encoder_hidden_states)

            #multi head
            encoder_hidden_states_query_proj = encoder_hidden_states_query_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            encoder_hidden_states_key_proj = encoder_hidden_states_key_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            encoder_hidden_states_value_proj = encoder_hidden_states_value_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)

            # Apply normalizations
            if attn.norm_added_q is not None:
                encoder_hidden_states_query_proj = attn.norm_added_q(encoder_hidden_states_query_proj)
            if attn.norm_added_k is not None:
                encoder_hidden_states_key_proj = attn.norm_added_k(encoder_hidden_states_key_proj)

            #concate
            query = torch.cat([encoder_hidden_states_query_proj, hidden_states_query_proj], dim=2)
            key = torch.cat([encoder_hidden_states_key_proj, key], dim=2)
            value = torch.cat([encoder_hidden_states_value_proj, value], dim=2)

        # attn1 = attention(q, k, v, pe=pe)
        if image_rotary_emb is not None:
            from diffusers.models.embeddings import apply_rotary_emb

            query = apply_rotary_emb(query, image_rotary_emb)
            key = apply_rotary_emb(key, image_rotary_emb)

        # attn1 = attention(q, k, v, pe=pe)
        hidden_states = F.scaled_dot_product_attention(
            query, key, value,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=False
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        if encoder_hidden_states is not None:
            encoder_hidden_states, hidden_states = (
                hidden_states[:, :encoder_hidden_states.shape[1]],
                hidden_states[:, encoder_hidden_states.shape[1]:],
            )

            # Linear projections
            hidden_states = attn.to_out[0](hidden_states)
            hidden_states = attn.to_out[1](hidden_states)
            encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

            # IP-Adapter processing
            ip_query = hidden_states_query_proj  # Use query projection directly like reference

            # Process IP states
            ip_key = self.to_k_ip(ip_hidden_states)
            ip_value = self.to_v_ip(ip_hidden_states)

            ip_key = ip_key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            ip_value = ip_value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

            current_ip_hidden_states = F.scaled_dot_product_attention(
                ip_query, ip_key, ip_value,
                attn_mask=None,
                dropout_p=0.0,
                is_causal=False
            )

            current_ip_hidden_states = current_ip_hidden_states.transpose(1, 2).reshape(
                batch_size, -1, attn.heads * head_dim
            )
            current_ip_hidden_states = current_ip_hidden_states.to(ip_query.dtype)

            # Add IP attention with scale
            hidden_states = hidden_states + self.scale[0] * current_ip_hidden_states

            return hidden_states, encoder_hidden_states
        else:
            return hidden_states