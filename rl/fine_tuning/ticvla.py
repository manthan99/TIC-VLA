import math
import os
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer, AutoProcessor
import time
import random
from concurrent.futures import ThreadPoolExecutor, Future, CancelledError
from threading import Event
from typing import Optional, List, Dict, Any, Tuple

# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def load_image(image_file, input_size=448, max_num=12):
    if not os.path.exists(image_file):
        raise FileNotFoundError(f"Image file not found: {image_file}")
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values


# -----------------------------------------------------------------------------
# VLA Modules
# -----------------------------------------------------------------------------

class CrossAttentionTransformer(nn.Module):
    """A single cross-attention + feed-forward block."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=input_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,  # (B, L, C)
        )

        self.ffn = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, input_dim),
        )

        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """Run one cross-attention + FFN block."""
        attn_out, _ = self.cross_attn(query=x, key=context, value=context)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))

        return x

class ActionExpertKVCache(nn.Module):
    """Produces a short-horizon chunked waypoint plan from KV cache context."""

    def __init__(
        self,
        input_dim: int,  # Dimension of image embeddings (hidden_size from VLM)
        hidden_dim: int,
        action_dim: int,
        num_layers: int = 3,
        num_chunks: int = 5,
        max_state_tokens: int = 16,
        kv_cache_feat_dim: int = 128,  # num_heads * head_dim (typically 2 * 64 = 128 for InternVL3-1B)
    ) -> None:
        super().__init__()

        self.num_chunks = num_chunks
        self.action_dim = action_dim
        self.max_state_tokens = max_state_tokens
        self.hidden_dim = hidden_dim

        # Down-project visual tokens (they are usually larger than 512)
        self.down_proj = nn.Linear(input_dim, hidden_dim)
        
        # Process KV cache: project directly from KV cache dimension to hidden_dim
        # KV cache from VLM: (B, num_heads, seq_len, head_dim) -> (B, seq_len, num_heads * head_dim)
        # Initialize kv_cache_proj here (not dynamically) for easier model loading
        self.kv_cache_proj = nn.Linear(kv_cache_feat_dim, hidden_dim)
        # Token dropout rate for KV cache (0.1 = drop 10% of tokens)
        self.kv_cache_token_dropout_rate = 0.1

        # discrete chunk embedding table 
        self.action_chunk_embed = nn.Embedding(num_chunks, hidden_dim)

        # Transformer cross-attention stack
        self.attention_layers = nn.ModuleList(
            [
                CrossAttentionTransformer(
                    input_dim=hidden_dim,
                    hidden_dim=4 * hidden_dim,
                    num_heads=8,
                    dropout=0.1,
                )
                for _ in range(num_layers)
            ]
        )

        # State encoder for robot low-dimensional states
        self.state_encoder = nn.Sequential(
            nn.Linear(1, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
        )

        # Learnable positional embeddings for scalar state sequence
        self.state_pos_embedding = nn.Embedding(max_state_tokens, hidden_dim)
        self.state_pos_dropout_rate = 0.5
        self.state_pos_dropout = nn.Dropout(self.state_pos_dropout_rate)

        # MLP head → predict each atomic action component
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ELU(),
            nn.Linear(hidden_dim // 2, action_dim),
        )

    def _process_kv_cache(self, past_key_values: Tuple[Tuple[torch.Tensor, torch.Tensor], ...]) -> torch.Tensor:
        """
        Process KV cache from VLM into a sequence of embeddings.

        Single path, assuming standard InternVL layout:
        values are shaped (B, num_heads, seq_len, head_dim).
        """
        if not past_key_values:
            # Return empty tensor with hidden_dim (will be set when kv_cache_proj is created)
            return torch.zeros(
                1,
                0,
                self.hidden_dim,
                device=next(self.parameters()).device,
                dtype=next(self.parameters()).dtype,
            )

        # Take values from the last layer
        _, last_layer_value = past_key_values[-1]
        if last_layer_value.dim() != 4:
            raise ValueError(
                f"Expected KV cache values of shape (B, num_heads, seq_len, head_dim), "
                f"got {last_layer_value.shape}. "
                f"If this triggers, print the shape once and adjust _process_kv_cache."
            )

        B, num_heads, seq_len, head_dim = last_layer_value.shape

        # (B, num_heads, seq_len, head_dim) -> (B, seq_len, num_heads * head_dim)
        value_reshaped = last_layer_value.permute(0, 2, 1, 3).contiguous()
        value_reshaped = value_reshaped.view(B, seq_len, num_heads * head_dim)

        # Verify dimension matches (kv_cache_proj is now initialized in __init__)
        feat_dim = value_reshaped.size(-1)  # num_heads * head_dim
        if feat_dim != self.kv_cache_proj.in_features:
            raise ValueError(
                f"KV cache feature dimension mismatch: expected {self.kv_cache_proj.in_features}, "
                f"got {feat_dim} (num_heads={num_heads}, head_dim={head_dim}). "
                f"Please update kv_cache_feat_dim in ActionExpertKVCache.__init__ to {feat_dim}."
            )
        
        # Project directly to hidden_dim
        value_reshaped = self.kv_cache_proj(value_reshaped)

        # Token-level dropout on KV cache
        if self.training and self.kv_cache_token_dropout_rate > 0:
            keep_prob = 1.0 - self.kv_cache_token_dropout_rate
            token_mask = torch.rand(
                B, seq_len, 1, device=value_reshaped.device, dtype=value_reshaped.dtype
            )
            token_mask = (token_mask < keep_prob).to(value_reshaped.dtype)
            value_reshaped = value_reshaped * token_mask

        return value_reshaped  # (B, seq_len, input_dim)

    def forward(
        self,
        image_embeds: torch.Tensor,
        state: torch.Tensor,
        kv_cache: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]] = None,
    ) -> torch.Tensor:
        """
        Return (B, T, 3) tensor - predicted waypoints relative to previous frame.
        
        Predicts waypoints (dx, dy, dtheta) relative to previous frame.
        
        Args:
            image_embeds: (B, L_img, H) image embeddings from current frame
            state: (B, S, 1) robot state tokens including velocity, displacement, and time delay
            kv_cache: Optional KV cache from VLM processing delayed frame context
        
        Returns:
            waypoints: (B, T, 3) tensor of waypoints [dx, dy, dtheta] relative to previous frame
        """
        B = image_embeds.size(0)

        # 1) Token preparation ------------------------------------------------
        image_embeds = self.down_proj(image_embeds)  # (B, L_img, hidden_dim)
        
        # Process KV cache if provided
        kv_embeds = None
        if kv_cache is not None:
            # _process_kv_cache already projects directly to hidden_dim (128 -> 512)
            kv_embeds = self._process_kv_cache(kv_cache)  # (B, L_kv, hidden_dim)
        
        state_embeds = self.state_encoder(state)  # (B, S, hidden_dim)
        state_len = state_embeds.size(1)
        if state_len > 0:
            pos_ids = torch.arange(state_len, device=state_embeds.device)
            pos_ids = pos_ids.unsqueeze(0).expand(B, -1)
            pos_emb = self.state_pos_embedding(pos_ids).to(dtype=state_embeds.dtype)
            state_embeds = self.state_pos_dropout(state_embeds + pos_emb)
        
        # Concatenate: KV cache (if available) + current image + state embeddings
        if kv_embeds is not None:
            hidden_states = torch.cat([kv_embeds, image_embeds, state_embeds], dim=1)  # (B, L_kv + L_img + S, hidden_dim)
        else:
            hidden_states = torch.cat([image_embeds, state_embeds], dim=1)  # (B, L_img + S, hidden_dim)

        # 2) Build action queries -------------------------------------------
        chunk_ids = torch.arange(self.num_chunks, device=hidden_states.device)
        chunk_emb = self.action_chunk_embed(chunk_ids)[None, ...].expand(B, -1, -1)  # (B, T, H)
        waypoint_queries = chunk_emb  # (B, T, H)

        # 3) Cross-attention stack ------------------------------------------
        for layer in self.attention_layers:
            waypoint_queries = layer(waypoint_queries, hidden_states)

        # 4) Directly predict waypoints relative to previous frame ------------
        waypoints = self.mlp(waypoint_queries)  # (B, T, 3) where 3 = [dx, dy, dtheta] relative to previous frame

        return waypoints

    def freeze_except_last_layer(self) -> None:
        """Freeze all action-expert weights except the final cross-attention block."""
        for param in self.parameters():
            param.requires_grad = False
        for param in self.attention_layers[-1].parameters():
            param.requires_grad = True


# -----------------------------------------------------------------------------
# TIC-VLA
# -----------------------------------------------------------------------------

class TICVLA(nn.Module):
    def __init__(
        self,
        model_path: str = 'InternVL3-1B',
        device: str = 'cuda:0',
        checkpoint_path: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.device = device
        
        # Async infrastructure for background KV cache generation
        self._executor: ThreadPoolExecutor = ThreadPoolExecutor(max_workers=1)
        self._kv_cache_future: Future | None = None
        self._kv_cache_start_time: float | None = None
        self._cancel_event: Event = Event()
        
        # Cache for latest KV cache (use previous while new one is generating)
        self._latest_kv_cache: Tuple[Tuple[torch.Tensor, torch.Tensor], ...] | None = None
        self._last_response: str | None = None  # Latest VLM response text (for reasoning)
        # Track when KV cache was generated in SIMULATION STEPS (for logging/debugging only)
        # Note: Delay calculation is done in behavior scripts, not here
        self._kv_cache_generation_step: int | None = None  # Simulation step/frame when current KV cache generation was STARTED
        self._kv_cache_generation_pose: dict | None = None  # Robot pose when KV cache generation started {'position': [...], 'quaternion': [...]}
        self._kv_cache_completion_step: int | None = None  # Step when the current generation COMPLETED (set when polling detects completion)
        self._has_ever_had_kv_cache: bool = False  # Track if we've ever successfully extracted KV cache
        # Hold a ready result if we need to delay releasing it to the policy
        self._pending_kv_cache_result: tuple[str | None, Tuple[Tuple[torch.Tensor, torch.Tensor], ...] | None] | None = None
        # Runtime tracking for logging
        self._last_vlm_runtime_sec: float | None = None

        # VLM and tokenizer
        self.vlm = AutoModel.from_pretrained(
            model_path,
            dtype=torch.bfloat16,
            use_flash_attn=False,
            trust_remote_code=True,
            device_map=device,
        ).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            use_fast=True,
            fix_mistral_regex=True,
        )
        self.processor = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=True,
            fix_mistral_regex=True,
        )

        # Freeze ALL VLM parameters (not just vision backbone)
        for param in self.vlm.parameters():
            param.requires_grad = False

        # Action expert that uses KV cache
        kv_num_heads = 2  # Fixed value for InternVL3-1B KV cache (NOT from VLM config)
        kv_head_dim = 64  # Fixed value for InternVL3-1B KV cache
        kv_cache_feat_dim = kv_num_heads * kv_head_dim  # 2 * 64 = 128
        self.action_expert = ActionExpertKVCache(
            input_dim=self.vlm.config.llm_config.hidden_size,
            hidden_dim=512,
            action_dim=2,
            num_layers=6,
            num_chunks=30,
            kv_cache_feat_dim=kv_cache_feat_dim,  # Initialize kv_cache_proj in __init__
        ).to(self.device).to(torch.bfloat16)
        self.action_expert.freeze_except_last_layer()

        # Load checkpoint if provided
        if checkpoint_path:
            self.load_checkpoint(checkpoint_path)

    def load_checkpoint(self, checkpoint_path: str):
        """Load checkpoint weights, handling both VLM and action_expert."""
        
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        state_dict = ckpt["state_dict"]
        
        # Remap keys from checkpoint format to model format
        new_state_dict = {}
        for k, v in state_dict.items():
            nk = k
            
            # Remove "model." prefix if present
            if nk.startswith("model."):
                nk = nk[6:]  # len("model.") = 6
            
            # Handle vlm_model.vlm. -> vlm. (these are duplicates, skip them)
            if nk.startswith("vlm_model.vlm."):
                continue  # Skip these, we'll use the vlm. prefix version
            
            # vlm. keys are already correct format for the model
            # action_expert. keys are already correct format for the model
            
            new_state_dict[nk] = v
        
        # Load with strict=False first to see what's missing/unexpected
        missing, unexpected = self.load_state_dict(new_state_dict)
        
        if missing:
            print(f"[WARNING] Missing keys ({len(missing)}): {missing[:10]}..." if len(missing) > 10 else f"[WARNING] Missing keys: {missing}")
        if unexpected:
            print(f"[WARNING] Unexpected keys ({len(unexpected)}): {unexpected[:10]}..." if len(unexpected) > 10 else f"[WARNING] Unexpected keys: {unexpected}")
        
        if not missing and not unexpected:
            print(f"[INFO] Checkpoint loaded successfully from {checkpoint_path}") 

    
    def forward(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Training forward pass with KV cache extraction from delayed frame context.

        Expects TIC-VLA collator batch keys:
        - delayed_input_ids: (B, N) input IDs for delayed frame context (past 4 images, COT, instruction)
        - delayed_attention_mask: (B, N)
        - delayed_pixel_values: (sum_tiles, C, H, W) images from delayed frame (past 4 images)
        - delayed_image_flags: (sum_tiles, 1)
        - delayed_num_tiles_per_sample: (B,) number of tiles per sample for delayed frame
        - current_image_paths: List[List[str]] paths to current frame's single closest image
        - robot_state: (B, 5) compact state tokens [vx, vy, yaw_speed, dx, dy]
                      where dx, dy is displacement from delayed time to current time
        - time_delay: (B,) scalar delay in seconds (2-8s)

        Returns:
        - predicted_waypoints: (B, T, 2) where 2 = [x, y]
        - language_loss: scalar tensor (dummy, since VLM is frozen)
        """
        # Ensure IMG_CONTEXT token id is set for InternVL forward() token replacement
        if getattr(self.vlm, 'img_context_token_id', None) is None:
            try:
                self.vlm.img_context_token_id = self.tokenizer.convert_tokens_to_ids('<IMG_CONTEXT>')
            except Exception:
                pass

        # Get delayed frame context for VLM (past 4 images, COT, instruction, trajectory)
        delayed_input_ids = batch['delayed_input_ids'].to(self.device)
        delayed_attention_mask = batch['delayed_attention_mask'].to(self.device)
        delayed_pixel_values = batch.get('delayed_pixel_values', None)
        delayed_image_flags = batch.get('delayed_image_flags', None)

        # Check if delayed_pixel_values exists and is non-empty
        # This happens when dataset's image_paths is empty (no delayed frame images collected)
        # Cause: dataset __getitem__ may not populate image_paths from history/current frames
        if delayed_pixel_values is None:
            raise RuntimeError(
                "delayed_pixel_values is None. This happens when the dataset has no delayed frame images. "
                "Check dataset __getitem__: image_paths should be populated from history frames. "
                "Each sample should have at least one delayed frame image (not just current frame)."
            )
        
        delayed_pixel_values = delayed_pixel_values.to(self.device).to(torch.bfloat16)
        
        # Check if tensor is empty (0 patches) - this shouldn't happen if dataset is correct
        if delayed_pixel_values.shape[0] == 0:
            raise RuntimeError(
                f"delayed_pixel_values is an empty tensor (shape: {delayed_pixel_values.shape}). "
                "This happens when delayed_image_paths exists but all images fail to load or produce 0 tiles. "
                "Check dataset: ensure delayed frame images exist and can be loaded."
            )
        
        # delayed_image_flags MUST exist if delayed_pixel_values exists (created by collator)
        if delayed_image_flags is None:
            raise RuntimeError(
                "delayed_image_flags is None but delayed_pixel_values exists. "
                "This is a collator bug - image_flags should always be created when pixel_values exist."
            )
        delayed_image_flags = delayed_image_flags.to(self.device)

        # Forward pass through VLM with delayed frame context to get KV cache
        # VLM processes: delayed frame's past 4 images + COT + instruction + trajectory
        with torch.no_grad():  # VLM is frozen, so no gradients needed
            # Build VLM forward kwargs - pixel_values is required by InternVL
            vlm_kwargs = {
                'input_ids': delayed_input_ids,
                'attention_mask': delayed_attention_mask,
                'pixel_values': delayed_pixel_values,
                'image_flags': delayed_image_flags,
                'return_dict': True,
            }
            
            try:
                vlm_kwargs['use_cache'] = True  # Enable KV cache
                vlm_kwargs['output_attentions'] = False
                vlm_outputs = self.vlm(**vlm_kwargs)
            except TypeError:
                # Fallback if use_cache is not supported
                vlm_outputs = self.vlm(**vlm_kwargs)
        
        # Extract past_key_values (KV cache) from VLM output
        past_key_values = getattr(vlm_outputs, 'past_key_values', None)
        
        # Handle DynamicCache format (transformers library)
        if past_key_values is not None:
            # Check if it's a DynamicCache object (from transformers.cache_utils)
            cache_type = type(past_key_values).__name__
            
            # Try to convert DynamicCache to tuple format
            if cache_type == 'DynamicCache':
                # DynamicCache has a 'layers' attribute that contains DynamicLayer objects
                if hasattr(past_key_values, 'layers') and len(past_key_values.layers) > 0:
                    # DynamicLayer uses 'keys' and 'values' (plural) - these are lists/tensors
                    # Need to access the actual key/value tensors from each layer
                    cache_tuple = []
                    for layer_idx, layer in enumerate(past_key_values.layers):
                        # DynamicLayer stores keys and values - need to get the actual tensors
                        # Use getattr to safely access attributes
                        layer_keys = getattr(layer, 'keys', None)
                        layer_values = getattr(layer, 'values', None)
                        
                        layer_key = None
                        layer_value = None
                        
                        if layer_keys is not None:
                            # keys might be a tensor, list, or other structure
                            if isinstance(layer_keys, torch.Tensor):
                                layer_key = layer_keys
                            elif isinstance(layer_keys, (list, tuple)) and len(layer_keys) > 0:
                                layer_key = layer_keys[0]  # Take first key tensor
                            elif hasattr(layer_keys, '__getitem__'):
                                try:
                                    layer_key = layer_keys[0]
                                except (IndexError, TypeError):
                                    pass
                        
                        if layer_values is not None:
                            # values might be a tensor, list, or other structure
                            if isinstance(layer_values, torch.Tensor):
                                layer_value = layer_values
                            elif isinstance(layer_values, (list, tuple)) and len(layer_values) > 0:
                                layer_value = layer_values[0]  # Take first value tensor
                            elif hasattr(layer_values, '__getitem__'):
                                try:
                                    layer_value = layer_values[0]
                                except (IndexError, TypeError):
                                    pass
                        
                        if layer_key is not None and layer_value is not None:
                            cache_tuple.append((layer_key, layer_value))
                        else:
                            print(
                                f"WARNING: Layer {layer_idx} - could not extract KV cache "
                                f"(keys type: {type(layer_keys)}, values type: {type(layer_values)})"
                            )
                            break
                    
                    if len(cache_tuple) > 0:
                        past_key_values = tuple(cache_tuple)
                    else:
                        print("WARNING: Could not extract any layers from DynamicCache!")
                        past_key_values = None
                elif hasattr(past_key_values, 'key_cache') and hasattr(past_key_values, 'value_cache'):
                    # Alternative structure: key_cache and value_cache lists
                    num_layers = len(past_key_values.key_cache) if past_key_values.key_cache else 0
                    if num_layers > 0:
                        cache_tuple = tuple(
                            (past_key_values.key_cache[i], past_key_values.value_cache[i])
                            for i in range(num_layers)
                        )
                        past_key_values = cache_tuple
                    else:
                        print("WARNING: DynamicCache has no layers!")
                        past_key_values = None
                else:
                    print(f"WARNING: Cannot extract KV cache from DynamicCache. Attributes: {[attr for attr in dir(past_key_values) if not attr.startswith('_')]}")
                    past_key_values = None
            elif isinstance(past_key_values, tuple):
                pass
            else:
                print(f"WARNING: Unknown past_key_values type: {type(past_key_values)}")
                past_key_values = None
        
        # If past_key_values is not available, fallback to extracting hidden states
        if past_key_values is None:
            # Fallback: use hidden states if available
            if hasattr(vlm_outputs, 'hidden_states') and vlm_outputs.hidden_states:
                # Use the last layer's hidden states as a proxy for KV cache
                hidden_states = vlm_outputs.hidden_states[-1]  # (B, seq_len, hidden_size)
                # Create a dummy past_key_values structure
                batch_size = hidden_states.shape[0]
                num_layers = getattr(self.vlm.config.llm_config, 'num_hidden_layers', 24)
                num_heads = getattr(self.vlm.config.llm_config, 'num_attention_heads', 16)
                head_dim = hidden_states.shape[-1] // num_heads
                seq_len = hidden_states.shape[1]
                
                # Reshape hidden states to match KV cache format
                hidden_reshaped = hidden_states.view(batch_size, seq_len, num_heads, head_dim)
                hidden_reshaped = hidden_reshaped.permute(0, 2, 1, 3)  # (B, num_heads, seq_len, head_dim)
                
                # Create dummy past_key_values
                past_key_values = tuple(
                    (hidden_reshaped.clone(), hidden_reshaped.clone())
                    for _ in range(num_layers)
                )
            else:
                # Last resort: create empty KV cache
                batch_size = delayed_input_ids.shape[0]
                past_key_values = tuple()
        
        # Language loss (dummy since VLM is frozen, but kept for compatibility)
        language_loss = vlm_outputs.loss if hasattr(vlm_outputs, 'loss') and vlm_outputs.loss is not None else torch.tensor(0.0, device=self.device)

        # State features and target waypoint conditioning
        # robot_state: (B, 5) = [vx, vy, yaw_speed, dx, dy]
        # where dx, dy is displacement from delayed time to current time
        robot_state = batch.get('robot_state').to(self.device).to(torch.bfloat16)  # (B, 5)

        # time delay (robot input)
        time_delay = (batch["time_delay"].to(self.device).to(torch.bfloat16)).unsqueeze(1)  # (B, 1)

        # concat robot state and time delay → (B, 5 + 1) = (B, 6)
        state = torch.cat([robot_state, time_delay], dim=1).unsqueeze(-1)  # (B, 6, 1)

        # Extract current frame image embeddings for waypoint prediction
        # Get the single closest image from current frame
        current_image_paths = batch.get('current_image_paths', [])  # List of lists: [[path1], [path2], ...]
        batch_size = delayed_input_ids.shape[0]
        
        if current_image_paths and len(current_image_paths) > 0:
            # Extract embeddings for each sample's current image
            current_image_embeds_list = []
            for paths in current_image_paths:
                pixel_values_current = load_image(paths[0], input_size=448, max_num=1).to(torch.bfloat16).to(self.device)
                with torch.no_grad():
                    img_embeds = self.vlm.extract_feature(pixel_values_current)  # (num_tiles, num_img_tokens, H)
                    img_embeds = img_embeds.reshape(-1, img_embeds.shape[-1])  # (total_tokens, H)
                current_image_embeds_list.append(img_embeds)
            
            # Pad to same length
            max_tokens = max(e.shape[0] for e in current_image_embeds_list) if current_image_embeds_list else 0
            if max_tokens > 0:
                padded_embeds = []
                for e in current_image_embeds_list:
                    if e.shape[0] < max_tokens:
                        pad = torch.zeros(max_tokens - e.shape[0], e.shape[1], device=e.device, dtype=e.dtype)
                        e = torch.cat([pad, e], dim=0)
                    padded_embeds.append(e)
                current_image_embeds = torch.stack(padded_embeds, dim=0)  # (B, L_img, H)
            else:
                current_image_embeds = torch.zeros(batch_size, 0, self.vlm.config.llm_config.hidden_size, device=self.device, dtype=torch.bfloat16)
        else:
            # Fallback: create empty embeddings if not provided
            current_image_embeds = torch.zeros(
                batch_size, 0, self.vlm.config.llm_config.hidden_size,
                device=self.device, dtype=torch.bfloat16
            )

        # Predict waypoints directly: action_expert outputs waypoints [x, y]
        # Uses KV cache (from delayed frame) + current image + state
        predicted_waypoints = self.action_expert(past_key_values, current_image_embeds, state)  # (B, T, 2)

        return predicted_waypoints, language_loss

    def _check_cancelled(self, cancel_event: Event | None = None) -> None:
        """Raise if a cancellation was requested for the current generation."""
        event = cancel_event if cancel_event is not None else self._cancel_event
        if event is not None and event.is_set():
            raise CancelledError("KV cache generation cancelled")

    @torch.inference_mode()
    def load_images(self, image_paths: list[str], input_size: int = 448, max_num: int = 1) -> torch.Tensor:
        pixel_values_list = []
        num_patches_list = []
        for p in image_paths:
            pv = load_image(p, input_size=input_size, max_num=max_num).to(torch.bfloat16).to(self.device)
            pixel_values_list.append(pv)
            num_patches_list.append(pv.shape[0])

        return pixel_values_list, num_patches_list


    def generate_and_extract_kv_cache(
        self,
        delayed_img_paths: list[str],
        instruction: str,
        generation_config: dict,
        previous_waypoints_text: str = "",
        robot_type: str = "mobile robot",
        cancel_event: Event | None = None,
    ) -> tuple[str, Tuple[Tuple[torch.Tensor, torch.Tensor], ...] | None]:
        """
        Generate text (for reasoning) and extract KV cache from delayed frame context (runs in background thread).
        Similar to waypoint version's generate(), but also extracts KV cache.
        
        Args:
            delayed_img_paths: List of image paths for delayed frame context
            instruction: Navigation instruction text
            generation_config: Generation config for text generation
            previous_waypoints_text: Optional text describing previous waypoints
            robot_type: Type of robot (e.g., "legged robot", "wheeled robot")
        
        Returns:
            Tuple of (response_text, kv_cache) or (response_text, None) if extraction fails
        """
        event = cancel_event if cancel_event is not None else self._cancel_event
        self._check_cancelled(event)

        # Load delayed images
        delayed_pixel_values_list, delayed_num_patches_list = self.load_images(
            delayed_img_paths, input_size=448, max_num=1
        )
        delayed_pixel_values = torch.cat(delayed_pixel_values_list, dim=0)
        self._check_cancelled(event)
        
        # Build prompt with robot type
        system_text = f"You are a {robot_type} assigned to perform navigation tasks.\n" + \
                      "You are provided with a video consisting of visual observations, including historical and current frames.\n"
        self.vlm.system_message = system_text
        
        user_text = f"The navigation instruction is: {instruction}"
        if previous_waypoints_text:
            user_text += f"\n{previous_waypoints_text}"
        user_text += "\nReturn the future target waypoints for the next 3s, 6s, and 9s in format: (x, y, theta). " + \
                     "Each waypoint represents the cumulative offset from the current position (total displacement over 3s, 6s, or 9s), " + \
                     "where x is positive for forward, y is positive for left, and theta is the heading angle in radians."
        
        prompt = ''.join([f'Frame {i}: <image>\n' for i in range(len(delayed_num_patches_list))]) + user_text

        # Generate text response (for reasoning) - this computes KV cache internally
        response = self.vlm.chat(
            self.tokenizer, 
            delayed_pixel_values, 
            prompt, 
            generation_config,
            history=None, 
            return_history=False, 
            num_patches_list=delayed_num_patches_list
        )
        self._check_cancelled(event)
        
        # Extract KV cache from FULL conversation (input + generated response)
        # Build full conversation messages including generated assistant response
        messages = [
            {'role': 'system', 'content': system_text},
            {'role': 'user', 'content': [{'type': 'image', 'image': p} for p in delayed_img_paths] + [{'type': 'text', 'text': user_text}]},
            {'role': 'assistant', 'content': [{'type': 'text', 'text': response}]}
        ]
        
        text_batch = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,  # No generation prompt needed - assistant response already included
        )
        
        # Build queries with image tokens
        # text_batch is a list of strings (one per sample in batch, but we only have 1 sample)
        if not isinstance(text_batch, list):
            text_batch = [text_batch]
        if len(text_batch) == 0:
            raise ValueError(f"[TIC-VLA] ERROR: text_batch is empty after apply_chat_template")
        
        IMG_START_TOKEN, IMG_END_TOKEN, IMG_CONTEXT_TOKEN = '<img>', '</img>', '<IMG_CONTEXT>'
        num_image_token = 256
        queries = []
        
        # Process each query in the batch (should be just 1 for inference)
        for query in text_batch:
            if not isinstance(query, str):
                raise ValueError(f"[TIC-VLA] ERROR: query is not a string: {type(query)}")
            
            # Replace <image> placeholders with tokens matching actual number of patches per image
            if '<image>' in query:
                # Replace one-by-one (same as collator)
                for tiles_for_image in delayed_num_patches_list:
                    if tiles_for_image > 0:
                        tokens_per_image = num_image_token * tiles_for_image
                        image_tokens = IMG_START_TOKEN + (IMG_CONTEXT_TOKEN * tokens_per_image) + IMG_END_TOKEN
                        query = query.replace('<image>', image_tokens, 1)
            else:
                # Prepend all images when placeholder is missing
                prepend = []
                for tiles_for_image in delayed_num_patches_list:
                    if tiles_for_image > 0:
                        tokens_per_image = num_image_token * tiles_for_image
                        prepend.append(IMG_START_TOKEN + (IMG_CONTEXT_TOKEN * tokens_per_image) + IMG_END_TOKEN)
                if len(prepend) > 0:
                    query = '\n'.join(prepend) + '\n' + query
            
            queries.append(query)
        
        if len(queries) == 0:
            raise ValueError(f"[TIC-VLA] ERROR: No queries built after processing text_batch")
        
        self.tokenizer.padding_side = 'left'
        tokenized = self.tokenizer(queries, return_tensors='pt', padding=True)
        delayed_input_ids = tokenized['input_ids'].to(self.device)
        delayed_attention_mask = tokenized['attention_mask'].to(self.device)
        delayed_image_flags = torch.ones(sum(delayed_num_patches_list), 1, dtype=torch.long, device=self.device)
        
        # Extract KV cache from FULL conversation (input + generated response)
        # This includes the VLM's reasoning in the KV cache
        with torch.no_grad():
            vlm_outputs = self.vlm(
                pixel_values=delayed_pixel_values,
                input_ids=delayed_input_ids,
                attention_mask=delayed_attention_mask,
                image_flags=delayed_image_flags,
                return_dict=True,
                use_cache=True,
            )
        
        past_key_values = vlm_outputs.past_key_values
        
        # Handle DynamicCache format
        kv_cache = None
        if past_key_values is not None and hasattr(past_key_values, 'layers'):
            kv_cache = tuple(
                (layer.keys, layer.values) 
                for layer in past_key_values.layers
            )
        
        return response, kv_cache

    def _handle_ready_kv_cache_result(
        self,
        response: str | None,
        kv_cache: Tuple[Tuple[torch.Tensor, torch.Tensor], ...] | None,
        current_step: int | None,
    ) -> bool:
        """Apply a finished VLM result, honoring any simulated step delay.
        
        Returns True if the result was consumed, False if it's stored for later.
        """
        self.simulated_vlm_delay_steps = 20

        if (
            self.simulated_vlm_delay_steps > 0
            and current_step is not None
            and self._kv_cache_generation_step is not None
        ):
            required_step = self._kv_cache_generation_step + self.simulated_vlm_delay_steps
            if current_step < required_step:
                # Hold onto the result until enough sim steps have passed
                remaining = required_step - current_step
                # print(
                #     f"[TIC-VLA] Holding VLM result for simulated latency: "
                #     f"waiting {remaining} more steps (target delay={self.simulated_vlm_delay_steps})"
                # )
                self._pending_kv_cache_result = (response, kv_cache)
                return False

        # Consume the result now
        self._pending_kv_cache_result = None
        self._last_response = response

        if current_step is not None:
            self._kv_cache_completion_step = current_step

        if kv_cache is not None:
            self._latest_kv_cache = kv_cache
            self._has_ever_had_kv_cache = True
        else:
            print(f"[TIC-VLA] WARNING: Generation completed but KV cache extraction returned None")

        if current_step is not None and self._kv_cache_generation_step is not None:
            if self._kv_cache_start_time is not None:
                self._last_vlm_runtime_sec = time.perf_counter() - self._kv_cache_start_time

        # Reset timer after consuming the result
        self._kv_cache_start_time = None
        return True

    def cancel_generation(self, reason: str = "") -> None:
        """Cancel any in-flight KV cache generation and drop pending results."""
        self._cancel_event.set()
        self._pending_kv_cache_result = None
        self._kv_cache_start_time = None
        self._kv_cache_completion_step = None
        self._kv_cache_generation_step = None
        self._kv_cache_generation_pose = None

        if self._kv_cache_future is not None:
            if not self._kv_cache_future.done():
                # Attempt to cancel if it hasn't started running yet
                self._kv_cache_future.cancel()
            self._kv_cache_future = None

    @torch.inference_mode()
    def _start_kv_cache_generation(
        self,
        delayed_img_paths: list[str],
        instruction: str,
        generation_config: dict,
        current_step: int | None = None,
        previous_waypoints_text: str = "",
        robot_type: str = "mobile robot",
        current_robot_pose: dict | None = None,
    ) -> None:
        """Start text generation and KV cache extraction in background thread if not already running.
        
        Starts a new generation if no generation is currently running.
        First call can use any number of images (even just 1).
        
        Args:
            current_robot_pose: Robot pose at the moment generation starts {'position': [...], 'quaternion': [...]}
                                Captured and stored when a new generation actually starts.
        """
        # If a finished result is still being delayed for release, wait and do not start a new generation.
        if self._pending_kv_cache_result is not None:
            return

        # Check if a generation is currently running
        # A generation is running if _kv_cache_future exists and is not done
        is_generation_running = (
            self._kv_cache_future is not None and 
            not self._kv_cache_future.done()
        )
        
        # Only start a new generation if no generation is currently running
        if not is_generation_running:
            cancel_event = Event()
            self._cancel_event = cancel_event
            self._kv_cache_start_time = time.perf_counter()
            # Track the step and pose when generation STARTS (for delay calculation and dx/dy)
            if current_step is not None:
                self._kv_cache_generation_step = current_step
                # Capture pose at the exact moment generation starts (matches the generation step)
                if current_robot_pose is not None:
                    pos = current_robot_pose.get('position', None)
                    quat = current_robot_pose.get('quaternion', None)
                    # Copy the pose values (they might be numpy arrays or lists)
                    if pos is not None:
                        pos = np.array(pos).copy() if not isinstance(pos, np.ndarray) else pos.copy()
                    if quat is not None:
                        quat = np.array(quat).copy() if not isinstance(quat, np.ndarray) else quat.copy()
                    self._kv_cache_generation_pose = {
                        'position': pos,
                        'quaternion': quat
                    }
            self._kv_cache_future = self._executor.submit(
                self.generate_and_extract_kv_cache,
                delayed_img_paths,
                instruction,
                generation_config,
                previous_waypoints_text,
                robot_type,
                cancel_event,
            )
        
    @torch.inference_mode()
    def _poll_kv_cache_future(self, current_step: int | None = None) -> None:
        """Poll for text generation and KV cache extraction completion, update latest cache and response.
        
        Args:
            current_step: Current simulation step/frame when KV cache is generated.
                         Used to track when the KV cache was created for delay calculation.
        """
        # If we already have a finished result waiting for simulated delay, try to release it
        if self._pending_kv_cache_result is not None:
            response, kv_cache = self._pending_kv_cache_result
            consumed = self._handle_ready_kv_cache_result(response, kv_cache, current_step)
            if not consumed:
                # Still waiting for the requested step delay
                return

        # Check if generation is running and completed
        if self._kv_cache_future is not None:
            if self._kv_cache_future.done():
                consumed = False
                try:
                    response, kv_cache = self._kv_cache_future.result()
                    consumed = self._handle_ready_kv_cache_result(response, kv_cache, current_step)
                except CancelledError:
                    print(f"[TIC-VLA] KV cache generation was cancelled.")
                except Exception as e:
                    import traceback
                    print(f"[TIC-VLA] ERROR in text generation/KV cache extraction: {e}")
                    print(f"[TIC-VLA] Traceback:")
                    traceback.print_exc()
                    # Don't set completion step if generation failed
                    # This ensures we don't track failed generations as completed
                finally:
                    self._kv_cache_future = None
                    # If the result is still pending due to simulated delay, keep start_time for logging
                    if consumed or self._pending_kv_cache_result is None:
                        self._kv_cache_start_time = None

    @torch.inference_mode()
    def predict(
        self,
        image_paths: list[str],
        instruction: str | None = None,
        robot_state: torch.Tensor | None = None,
        history: Optional[List[Dict[str, Any]]] = None,
        current_timestamp: Optional[float] = None,
        previous_waypoints_text: Optional[str] = None,
    ) -> tuple[str, torch.Tensor, str]:
        """
        Inference pipeline producing VLM guidance text and decoded waypoints using KV cache.
        
        For inference, we use the same images for both delayed frame context (VLM) and 
        current frame (waypoint prediction). In real deployment, delayed frame would come from
        history and current frame would be the latest observation.
        
        Args:
            image_paths: List of image paths (used for both delayed context and current frame)
            instruction: Navigation instruction text
            robot_state: Current robot state tensor [vx, vy, yaw_speed, dx, dy]
                        where dx, dy is displacement from delayed time to current time
            history: Optional history (not used in simplified inference)
            current_timestamp: Optional timestamp (not used in simplified inference)
            previous_waypoints_text: Optional past trajectory text (matches training format)
        
        Returns:
            response: Generated assistant response
            waypoints: Predicted waypoints (B, T, 2) where 2 = [x, y]
            prompt: Full prompt string sent to model (for visualization)
        """
        if not instruction:
            raise ValueError("instruction cannot be None or empty. Must provide a valid instruction.")
        # 1) Load camera frames for VLM (delayed frame context) ------------
        # In training, this would be past 4 images from delayed frame
        # For inference, we use the provided images as delayed context
        if not image_paths:
            raise ValueError("image_paths cannot be empty. At least one image path is required.")
        
        pixel_values_list, num_patches_list = self.load_images(
            image_paths=image_paths,
            input_size=448,
            max_num=1,
        )
        
        if not pixel_values_list:
            raise ValueError(f"Failed to load any images from paths: {image_paths}")
        
        if not num_patches_list or sum(num_patches_list) == 0:
            raise ValueError(f"No image patches loaded. num_patches_list: {num_patches_list}")

        # 2) Prepare input for VLM (delayed frame context) ------------------
        pixel_values = torch.cat(pixel_values_list, dim=0)
        total_patches = pixel_values.shape[0]
        
        # 2) Generate assistant response first (COT + guidance waypoint) ------------
        # Build prompt for generation (same format as training - MUST match dataset format)
        system_text = "You are a physical mobile robot assigned to perform navigation tasks.\n" + \
                      "You are provided with a video consisting of visual observations, including historical and current frames.\n"
        self.vlm.system_message = system_text
        
        user_text = f"The navigation instruction is: {instruction}"
        # Include past trajectory if provided (matches training format)
        if previous_waypoints_text:
            user_text += f"\n{previous_waypoints_text}"
        user_text += "\nReturn the future target waypoints for next 5s and 10s in format: (x, y, theta). Each waypoint represents the cumulative offset from the current position (total displacement over 5s or 10s), where x is positive for forward, y is positive for left, and theta is the heading angle in radians."
        
        # Build prompt string for VLM chat (same as ticvla.py generate method)
        # This is used for generation
        generation_prompt = ''.join([f'Frame {i}: <image>\n' for i in range(len(num_patches_list))]) + user_text
        
        # Store full prompt text (system + user with past trajectory) for return - this is what actually gets sent to model
        full_prompt_text = f"SYSTEM:\n{system_text}\n\nUSER:\n{user_text}"
        
        # Generate assistant response (this will create KV cache internally)
        generation_config = dict(max_new_tokens=200, do_sample=True, temperature=0.1, top_p=0.1, top_k=1)
        with torch.no_grad():
            generated_response = self.vlm.chat(
                self.tokenizer, 
                pixel_values, 
                generation_prompt,  # Use generation_prompt (includes Frame X: <image>)
                generation_config,
                history=None, 
                return_history=False, 
                num_patches_list=num_patches_list
            )
        
        # 3) Build full conversation (user + generated assistant) to extract KV cache
        messages = [
            {'role': 'system', 'content': system_text},
            {'role': 'user', 'content': [{'type': 'image', 'image': p} for p in image_paths] + [{'type': 'text', 'text': user_text}]}
        ]
        
        text_batch = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        
        # Handle list return from apply_chat_template
        if isinstance(text_batch, list):
            text_batch = text_batch[0] if len(text_batch) > 0 else ""
        
        # Build queries with image tokens - must match actual number of patches
        IMG_START_TOKEN, IMG_END_TOKEN, IMG_CONTEXT_TOKEN = '<img>', '</img>', '<IMG_CONTEXT>'
        # Calculate num_image_token from model config (same as collator's _compute_num_image_token)
        num_image_token = 256  # Default for InternVL
        
        # Replace <image> placeholders with tokens matching actual number of patches per image
        query = text_batch
        if '<image>' in query:
            # Replace one-by-one (same as collator)
            for tiles_for_image in num_patches_list:
                if tiles_for_image > 0:
                    tokens_per_image = num_image_token * tiles_for_image
                    image_tokens = IMG_START_TOKEN + (IMG_CONTEXT_TOKEN * tokens_per_image) + IMG_END_TOKEN
                    query = query.replace('<image>', image_tokens, 1)
        else:
            # Prepend all images when placeholder is missing
            prepend = []
            for tiles_for_image in num_patches_list:
                if tiles_for_image > 0:
                    tokens_per_image = num_image_token * tiles_for_image
                    prepend.append(IMG_START_TOKEN + (IMG_CONTEXT_TOKEN * tokens_per_image) + IMG_END_TOKEN)
            if len(prepend) > 0:
                query = '\n'.join(prepend) + '\n' + query

        self.tokenizer.padding_side = 'left'
        tokenized = self.tokenizer([query], return_tensors='pt', padding=True)
        input_ids = tokenized['input_ids'].to(self.device)
        attention_mask = tokenized['attention_mask'].to(self.device)
        
        # image_flags: (total_patches, 1) tensor of ones - MUST match number of image patches
        image_flags = torch.ones(total_patches, 1, dtype=torch.long, device=self.device)

        # 4) Forward pass through VLM with full conversation to get KV cache
        with torch.no_grad():
            vlm_outputs = self.vlm(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                image_flags=image_flags,
                return_dict=True,
                use_cache=True,
            )
        
        past_key_values = vlm_outputs.past_key_values
        
        # Convert DynamicCache to tuple format: DynamicCache.layers[i].keys/values are tensors
        if past_key_values is not None and hasattr(past_key_values, 'layers'):
            past_key_values = tuple(
                (layer.keys, layer.values) 
                for layer in past_key_values.layers
            )

        # 4) Extract current frame image embeddings (single closest image) ---
        # In training, this would be the current frame's single closest image
        # For inference, we use the first image from image_paths as current frame
        if image_paths and len(image_paths) > 0:
            current_image_path = image_paths[0]  # Use first image as current frame
            pixel_values_current = load_image(current_image_path, input_size=448, max_num=1).to(torch.bfloat16).to(self.device)
            with torch.no_grad():
                current_image_embeds = self.vlm.extract_feature(pixel_values_current)  # (num_tiles, num_img_tokens, H)
                current_image_embeds = current_image_embeds.reshape(-1, current_image_embeds.shape[-1])  # (total_tokens, H)
                current_image_embeds = current_image_embeds.unsqueeze(0)  # (1, L_img, H)
        else:
            current_image_embeds = torch.zeros(1, 0, self.vlm.config.llm_config.hidden_size, device=self.device, dtype=torch.bfloat16)

        # 5) Prepare robot state tokens --------------------------------------
        # robot_state should be [vx, vy, yaw_speed, dx, dy] (5 dims)
        # If not provided or shorter, pad with zeros
        if robot_state is None:
            robot_state_tensor = torch.zeros(5, device=self.device, dtype=torch.bfloat16)
        else:
            if not torch.is_tensor(robot_state):
                robot_state = torch.tensor(robot_state)
            robot_state_tensor = robot_state.to(self.device, dtype=torch.bfloat16).view(-1)

        time_delay = torch.zeros(1, device=self.device, dtype=torch.bfloat16)
        
        # Concat robot state and time delay → (6,)
        state = torch.cat([robot_state_tensor, time_delay], dim=0).unsqueeze(0).unsqueeze(-1)  # (1, 6, 1)

        # 6) Decode trajectory using KV cache + current image + state --------
        waypoints = self.action_expert(past_key_values, current_image_embeds, state)  # (1, T, 2)

        # Return generated response, waypoints, and full prompt text (with past trajectory) for visualization
        return generated_response, waypoints, full_prompt_text

    @torch.inference_mode()
    def predict_async(
        self,
        image_paths: list[str],
        instruction: str | None = None,
        robot_state: torch.Tensor | None = None,
        current_step: int | None = None,
        current_robot_pose: dict | None = None,
        time_delay: float = 0.0,
        previous_waypoints_text: str = "",
        delayed_image_paths: list[str] | None = None,
        robot_type: str = "mobile robot",
    ) -> tuple[Optional[str], torch.Tensor]:
        """
        Non-blocking, polling-based inference step using KV cache.

        - Starts text generation (for reasoning) and KV cache extraction in background from delayed frame context (non-blocking)
        - Polls for completion and updates latest KV cache and response text when ready
        - Always uses latest available KV cache (from previous call if new one not ready)
        - Decodes actions using KV cache, current image embeddings, robot state, and time delay

        Args:
            image_paths: List of image file paths for current frame
            instruction: Navigation instruction text
            robot_state: Robot state tensor [vx, vy, yaw_speed, dx, dy]
                        where dx, dy is displacement from delayed time to current time
            current_step: Current simulation step/frame number.
                         Used for KV cache generation tracking and fallback delay calculation.
            current_robot_pose: Current robot pose (for compatibility with waypoint version, not used in KV cache version).
            time_delay: Time delay in seconds between current frame and second-to-last inference start frame.
                       Should be provided by behavior scripts: delay_time = (current_frame - ref_inference_start_frame) * (1.0 / 30.0).
                       Calculated from frame difference, not simulation time, for consistency.
            previous_waypoints_text: Optional text describing previous waypoints (for prompt context).
            delayed_image_paths: List of image paths for delayed frame context (past 4 images).
                                If None, uses current images as fallback.
            robot_type: Type of robot (e.g., "legged robot", "wheeled robot")

        Returns (response_or_none, waypoints, vlm_generation_start_step, kv_cache_available, vlm_generation_start_pose):
        - response_or_none: Latest VLM response text (with reasoning) if one has completed, else None
        - waypoints: (1, T, 2) decoded waypoints [x, y]
        - vlm_generation_start_step: Frame number when VLM generation started (None if no new generation started this call)
        - kv_cache_available: True if KV cache is available (at least one generation has completed), False otherwise
        - vlm_generation_start_pose: Robot pose when generation started {'position': [...], 'quaternion': [...]} (None if no new generation started)
        """
        if instruction is None:
            instruction = "Move forward safely and efficiently."

        # 1) Poll for previous KV cache completion FIRST (before starting new generation)
        self._poll_kv_cache_future(current_step)
        
        # 2) Start text generation and KV cache extraction in background (non-blocking)
        # VLM uses 4 images from generation time: current frame + 3 earlier frames (each ~3s apart)
        # These are the "delayed" images - the context when KV cache was generated
        # image_paths contains: [oldest (-9s), -6s, -3s, current] after reversal in behavior script
        # If delayed_image_paths is provided, use it; otherwise use image_paths (which contains 4 sampled images)
        delayed_img_paths = delayed_image_paths if delayed_image_paths is not None else image_paths
        generation_config = dict(max_new_tokens=200, do_sample=True)
        
        # Track VLM generation start step before calling (to detect if new generation started)
        prev_vlm_generation_step = self._kv_cache_generation_step
        
        # Pass current_robot_pose so it can be captured when generation actually starts
        self._start_kv_cache_generation(
            delayed_img_paths,
            instruction,
            generation_config,
            current_step,
            previous_waypoints_text,
            robot_type,
            current_robot_pose=current_robot_pose
        )
        
        # Check if a new VLM generation started (for behavior script tracking)
        vlm_generation_start_step = None
        vlm_generation_start_pose = None
        if self._kv_cache_generation_step != prev_vlm_generation_step:
            # New VLM generation started - return the step and pose that were captured
            vlm_generation_start_step = self._kv_cache_generation_step
            vlm_generation_start_pose = self._kv_cache_generation_pose
        # Note: vlm_generation_start_step is None if no new generation started (previous one still running)
        
        # 5) Use latest KV cache (from previous call if new one not ready yet)
        # For first generation, wait until KV cache is available (don't use empty cache)
        kv_cache_to_use = self._latest_kv_cache
        if kv_cache_to_use is None:
            # On first generation, wait for KV cache to be available
            # But if we've had KV cache before and now it's None, that's an error
            if self._has_ever_had_kv_cache:
                # ERROR: We've had KV cache before, but now it's None - extraction must be failing
                raise RuntimeError(f"[TIC-VLA] ERROR: No KV cache available at step={current_step} "
                                 f"but we've had KV cache before. Generation started at step={self._kv_cache_generation_step}. "
                                 f"This indicates KV cache extraction is failing. Check error logs above.")
            else:
                # First generation - wait for KV cache to be available (blocking)
                # Don't use empty cache - wait until generation completes
                if self._kv_cache_future is not None:
                    # Wait for generation to complete (blocking - returns immediately if already done)
                    try:
                        response, kv_cache = self._kv_cache_future.result(timeout=30.0)  # 30 second timeout
                        self._last_response = response
                        if kv_cache is not None:
                            self._latest_kv_cache = kv_cache
                            self._has_ever_had_kv_cache = True
                            kv_cache_to_use = kv_cache
                        else:
                            raise RuntimeError(f"[TIC-VLA] First generation completed but KV cache extraction returned None")
                    except Exception as e:
                        import traceback
                        print(f"[TIC-VLA] ERROR waiting for first generation: {e}")
                        traceback.print_exc()
                        raise RuntimeError(f"[TIC-VLA] Failed to get KV cache from first generation: {e}")
                    finally:
                        self._kv_cache_future = None
                        self._kv_cache_start_time = None
                else:
                    # Generation should be running, but future is None - this shouldn't happen
                    raise RuntimeError(f"[TIC-VLA] First generation: No KV cache and no generation future. "
                                     f"This should not happen - generation should have been started.")

        # 6) Extract current frame image embeddings (use current frame image, not delayed images)
        # Action expert uses the current frame's single closest image (latest observation)
        # image_paths contains 4 images: [oldest, -6s, -3s, current]
        # So the current frame is the LAST image (most recent)
        current_image_path = image_paths[-1]  # Use last image (current/most recent frame)
        pixel_values_current = load_image(current_image_path, input_size=448, max_num=1).to(torch.bfloat16).to(self.device)
        with torch.no_grad():
            image_embeds = self.vlm.extract_feature(pixel_values_current)  # (num_tiles, num_img_tokens, H)
            image_embeds = image_embeds.reshape(-1, image_embeds.shape[-1])  # (total_tokens, H)
            image_embeds = image_embeds.unsqueeze(0)  # (1, L_img, H)


        # 7) Build state with robot state and time delay
        time_delay_tensor = torch.tensor([time_delay], device=self.device, dtype=image_embeds.dtype)

        robot_state = robot_state.to(self.device, dtype=image_embeds.dtype)

        # KV cache version uses robot_state (5 dims: vx, vy, yaw_speed, dx, dy) + time_delay (1 dim) = 6 dims total
        state = torch.cat([robot_state, time_delay_tensor], dim=0).view(1, -1, 1)  # (1, 6, 1)

        # 8) Decode waypoints using KV cache (returns waypoints [x, y])
        waypoints = self.action_expert(kv_cache_to_use, image_embeds, state)  # (1, T, 2) [x, y]

        # Return whether KV cache is available (for first inference completion check)
        kv_cache_available = self._has_ever_had_kv_cache and self._latest_kv_cache is not None

        return self._last_response, waypoints, vlm_generation_start_step, kv_cache_available, vlm_generation_start_pose
