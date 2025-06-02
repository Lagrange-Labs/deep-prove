import torch
import re
import json
from transformers import GPT2Model, GPT2Tokenizer
from transformers import AutoModelForCausalLM
from transformers.modeling_utils import Conv1D
import os

torch.set_printoptions(precision=6, sci_mode=False)

# Load tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
# model = GPT2Model.from_pretrained("gpt2", output_hidden_states=True)
model = AutoModelForCausalLM.from_pretrained("sshleifer/tiny-gpt2")

model.eval()

# Input
input_text = "Hello"
input_ids = tokenizer.encode(input_text, return_tensors="pt")

# Run through model to get reference
with torch.no_grad():
    outputs = model(input_ids, output_hidden_states=True)
    inputs_embeds = outputs.hidden_states[0] # This is wte(input_ids) + wpe(position_ids)

    # Explicitly calculate token and positional embeddings
    input_seq_length = input_ids.size(1)
    position_ids = torch.arange(0, input_seq_length, dtype=torch.long, device=input_ids.device).unsqueeze(0)
    
    token_embeds_only = model.transformer.wte(input_ids)
    pos_embeds_only = model.transformer.wpe(position_ids)

    # Sanity check: their sum should be the inputs_embeds
    assert torch.allclose(inputs_embeds, token_embeds_only + pos_embeds_only, atol=1e-6), \
        "Sum of explicit token and positional embeddings does not match hidden_states[0]"

# ---- Manual forward through full first block ---- #
with torch.no_grad():
    layer = model.transformer.h[0]
    hidden_size = model.config.hidden_size # For splitting

    # LayerNorm before attention
    ln1_out = layer.ln_1(inputs_embeds)
    print(f"ℹ️ INPUT to QKV (ln1_out) shape {ln1_out.shape} => {ln1_out}")

    # QKV weights and biases from c_attn layer
    c_attn_module = layer.attn.c_attn
    
    # Print original fused weights and biases
    print(f"ℹ️ c_attn module FULL weights (c_attn_module.weight) shape: {c_attn_module.weight.shape}:\n{c_attn_module.weight.data}")
    print(f"ℹ️ c_attn module FULL bias (c_attn_module.bias) shape: {c_attn_module.bias.shape}:\n{c_attn_module.bias.data}")

    # Split weights and biases for Q, K, V
    # c_attn_module.weight is [hidden_size, 3 * hidden_size]
    # c_attn_module.bias is [3 * hidden_size]
    W_q, W_k, W_v = c_attn_module.weight.split(hidden_size, dim=1)
    b_q, b_k, b_v = c_attn_module.bias.split(hidden_size, dim=0)

    print(f"ℹ️ W_q (weight for Q) shape: {W_q.shape} => {W_q.data.flatten().tolist()}")
    print(f"ℹ️ W_k (weight for K) shape: {W_k.shape} => {W_k.data.flatten().tolist()}")
    print(f"ℹ️ W_v (weight for V) shape: {W_v.shape} => {W_v.data.flatten().tolist()}")
    print(f"ℹ️ b_q (bias for Q) shape: {b_q.shape} => {b_q.data.flatten().tolist()}")
    print(f"ℹ️ b_k (bias for K) shape: {b_k.shape} => {b_k.data.flatten().tolist()}")
    print(f"ℹ️ b_v (bias for V) shape: {b_v.shape} => {b_v.data.flatten().tolist()}")

    # Calculate Q, K, V separately (textbook style)
    # ln1_out: [batch, seq_len, hidden_size]
    # W_q, W_k, W_v: [hidden_size, hidden_size]
    # b_q, b_k, b_v: [hidden_size]
    q_before_heads = torch.matmul(ln1_out, W_q) + b_q
    k_before_heads = torch.matmul(ln1_out, W_k) + b_k
    v_before_heads = torch.matmul(ln1_out, W_v) + b_v
    
    # The print for "Fused QKV tensor" is removed as we are not calculating it that way anymore.
    # The following print will now show the Q tensor as calculated by the separate matmul.
    print(f"ℹ️ Q_before_heads (ln1_out @ W_q + b_q) shape: {q_before_heads.shape}:\n{q_before_heads}")
    # Optionally print k_before_heads and v_before_heads if needed:
    # print(f"ℹ️ K_before_heads (ln1_out @ W_k + b_k) shape: {k_before_heads.shape}:\n{k_before_heads}")
    # print(f"ℹ️ V_before_heads (ln1_out @ W_v + b_v) shape: {v_before_heads.shape}:\n{v_before_heads}")

    def split_heads(x):
        B, T, C = x.size()
        H = model.config.n_head
        return x.view(B, T, H, C // H).transpose(1, 2)

    def merge_heads(x):
        B, H, T, D = x.size()
        return x.transpose(1, 2).contiguous().view(B, T, H * D)

    q = split_heads(q_before_heads)
    k = split_heads(k_before_heads)
    v = split_heads(v_before_heads)

    print(f"ℹ️ Q tensor (q) shape: {q.shape}:\n{q}")
    print(f"ℹ️ K tensor (k) shape: {k.shape}:\n{k}")
    print(f"ℹ️ V tensor (v) shape: {v.shape}:\n{v}")

    # Attention
    attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (q.size(-1) ** 0.5)
    mask = torch.tril(torch.ones_like(attn_scores[0, 0])).unsqueeze(0).unsqueeze(0)
    attn_scores = attn_scores.masked_fill(mask == 0, float("-inf"))

    attn_weights = torch.softmax(attn_scores, dim=-1)
    attn_output = torch.matmul(attn_weights, v)
    attn_output = merge_heads(attn_output)

    # Output proj + dropout + residual
    attn_output_proj = layer.attn.c_proj(attn_output)
    attn_output_proj = torch.nn.functional.dropout(attn_output_proj, p=model.config.resid_pdrop, training=model.training)
    residual_attn = inputs_embeds + attn_output_proj

    # LayerNorm before MLP
    ln2_out = layer.ln_2(residual_attn)

    # FFN (c_fc -> GELU -> c_proj -> dropout)
    ffn_intermediate = layer.mlp.c_fc(ln2_out)
    ffn_activated = layer.mlp.act(ffn_intermediate)
    ffn_output_proj = layer.mlp.c_proj(ffn_activated)
    ffn_output_proj = torch.nn.functional.dropout(ffn_output_proj, p=model.config.resid_pdrop, training=model.training)


    manual_output = residual_attn + ffn_output_proj

# Validate
automated_output = outputs.hidden_states[1]

if not torch.allclose(manual_output, automated_output, atol=1e-4):
    max_diff = (manual_output - automated_output).abs().max()
    print(f"❌ Mismatch. Max diff: {max_diff.item():.6f}")
    raise ValueError("Mismatch between manual and automatic full block output.")

# Dump
def to_list(t): return t.squeeze().flatten().tolist()

output_json = {
    "token": input_text,
    "input_ids": input_ids.squeeze().tolist(),
    "token_embeds_only": to_list(token_embeds_only),
    "pos_embeds_only": to_list(pos_embeds_only),
    "inputs_embeds": to_list(inputs_embeds),
    "ln1_out": to_list(ln1_out),
    "ln2_out": to_list(ln2_out),
    "q": to_list(q_before_heads),
    "k": to_list(k_before_heads),
    "v": to_list(v_before_heads),
    "attn_scores": to_list(attn_scores),
    "attn_weights": to_list(attn_weights),
    "attn_output_proj": to_list(attn_output_proj),
    "attn_output": to_list(attn_output),
    "residual_attn": to_list(residual_attn),
    "ffn_up": to_list(ffn_intermediate),
    "ffn_activated": to_list(ffn_activated),
    "ffn_output_proj": to_list(ffn_output_proj),
    "manual_output": to_list(manual_output),
    "automated_output": to_list(automated_output)
}

# Save the debug tensor outputs
output_debug_fname = os.path.join(args.output_dir, "gpt2_debug_output.json")
with open(output_debug_fname, "w") as f:
    json.dump(output_json, f, indent=2)
print(f"✅ Debug outputs written to {output_debug_fname}")

import json
from collections import OrderedDict

state_dict = model.state_dict()

# Generic metadata keys for the JSON export
export_metadata = {
    "model_name": model.config.name_or_path,
    "model_type": model.config.model_type,
    "hidden_dim": model.config.hidden_size,
    "embedding_dim": model.config.hidden_size,  # For GPT-2, n_embd is hidden_size
    "num_hidden_layers": model.config.n_layer,
    "num_attention_heads": model.config.n_head,
    "vocab_size": model.config.vocab_size,
    "norm_epsilon": model.config.layer_norm_epsilon,
    "max_seq_len": model.config.n_positions,
}

# GGUF-style JSON output
export = {
    "metadata": export_metadata,
    "tensors": {} 
}

# Parse blocks (e.g., transformer.h.0.attn.c_attn.weight)
block_re = re.compile(r"transformer\.h\.(\d+)\.(.+)")

# Mapping for common non-block tensors from HF names to canonical names
# (closer to GGUF where applicable)
hf_to_canonical_map = {
    "transformer.wte.weight": "token_embd.weight",
    "transformer.wpe.weight": "position_embd.weight",
    "transformer.ln_f.weight": "output_norm.weight",
    "transformer.ln_f.bias": "output_norm.bias",
    "lm_head.weight": "output.weight"
}

# List of HF tensor name suffixes (within a block) that come from Conv1D
# These are relative to the block, e.g., "h.0.attn.c_attn.weight" -> "attn.c_attn.weight"
# We no longer need to transpose them, but we might need to identify them for special formatting (like c_attn)
conv1d_module_suffixes = [
    "attn.c_attn.weight", "attn.c_attn.bias",
    "attn.c_proj.weight", "attn.c_proj.bias",
    "mlp.c_fc.weight",    "mlp.c_fc.bias",
    "mlp.c_proj.weight",  "mlp.c_proj.bias",
]

# We also need to check biases and other weights that *shouldn't* be Conv1D weights
# These are suffixes of tensors that are part of layers we expect NOT to be Conv1D weights
# (e.g., LayerNorm weights/biases)
non_conv1d_module_tensor_suffixes = [
    "ln_1.weight", "ln_1.bias",
    "ln_2.weight", "ln_2.bias",
]

for name, tensor in state_dict.items():
    
    processed_tensor = tensor.cpu()
    # Initialize current_shape with the original shape. It might be overridden.
    current_shape = list(processed_tensor.shape)
    current_data_list = None

    match_block = block_re.match(name)
    if match_block:
        layer_id_str = match_block.group(1)
        hf_subname = match_block.group(2) # e.g., "attn.c_attn.weight"

        if hf_subname in conv1d_module_suffixes:
            try:
                module_path_parts = hf_subname.split('.')[:-1]
                current_module_for_type_check = model.transformer.h[int(layer_id_str)]
                for part in module_path_parts:
                    current_module_for_type_check = getattr(current_module_for_type_check, part)
                assert isinstance(current_module_for_type_check, Conv1D), \
                    f"Error: Expected module for {name} (subname: {hf_subname}) to be Conv1D, but got {type(current_module_for_type_check)}."

                if hf_subname == "attn.c_attn.weight":
                    # Fused QKV weights: PyTorch shape [hidden_size, 3 * hidden_size]
                    # Components W_q, W_k, W_v are [hidden_size, hidden_size] ([in, out])
                    H = processed_tensor.shape[0] # hidden_size
                    W_q_orig, W_k_orig, W_v_orig = processed_tensor.split(H, dim=1)

                    # Transpose each component to [out, in] before flattening
                    current_data_list = W_q_orig.flatten().tolist() + \
                                        W_k_orig.flatten().tolist() + \
                                        W_v_orig.flatten().tolist()
                    # Shape for JSON is 1D concatenated data
                    current_shape = [len(current_data_list)]
                    print(f"ℹ️ Exporting {name} (as attn_qkv.weight): Original Conv1D shape {list(processed_tensor.shape)}. Exporting concatenated W_q.T, W_k.T, W_v.T data as 1D array, shape {current_shape}.")

                elif hf_subname.endswith(".weight"): # Other Conv1D weights (e.g., mlp.c_fc, mlp.c_proj, attn.c_proj)
                    # These are [in_features, out_features]. Transpose to [out_features, in_features].
                    transposed_tensor = processed_tensor.T
                    current_shape = list(transposed_tensor.shape)
                    current_data_list = transposed_tensor.numpy().flatten().tolist()
                    print(f"ℹ️ Exporting {name}: Original Conv1D shape {list(processed_tensor.shape)}, transposing to {current_shape} for export.")
                
                # For Conv1D biases (hf_subname.endswith(".bias")):
                # No special handling here, they fall through to the `current_data_list is None` block.
                # current_shape remains original, current_data_list will be original flattened data.

            except AttributeError:
                print(f"⚠️ Warning: Could not access module for {name} to assert its type for Conv1D check.")
            except AssertionError as e:
                print(f"❌ Critical Error for {name}: {e}")
                raise e
        
        elif hf_subname in non_conv1d_module_tensor_suffixes:
            # For LayerNorms, etc. No transpose.
            # Fall through to the `current_data_list is None` block.
            try:
                module_name_in_block = hf_subname.rsplit('.', 1)[0]
                current_module_for_type_check = model.transformer.h[int(layer_id_str)]
                for part in module_name_in_block.split('.'):
                    current_module_for_type_check = getattr(current_module_for_type_check, part)
                assert not isinstance(current_module_for_type_check, Conv1D), \
                    f"Error: Expected module for {name} (module: {module_name_in_block}) NOT to be Conv1D, but it is ({type(current_module_for_type_check)})."
            except AttributeError:
                print(f"⚠️ Warning: Could not access module for {name} (subname: {hf_subname}) to assert its type properties (non-Conv1D check).")
            except AssertionError as e:
                print(f"❌ Critical Error for {name}: {e}")
                raise e
    
    # Fallback for tensors not specially handled above (e.g., biases, embeddings, LayerNorms)
    if current_data_list is None:
        current_data_list = processed_tensor.numpy().flatten().tolist()
        # current_shape was already set to list(processed_tensor.shape) and is correct here.

    if all(x == 0.0 for x in current_data_list):
        print(f"⚠️ Warning: Tensor '{name}' (Exported Shape: {current_shape}) consists entirely of zeros.")
    
    tensor_data = {
        "shape": current_shape,
        "data": current_data_list 
    }

    # Map to canonical names and store in export dictionary
    if name in hf_to_canonical_map:
        export["tensors"][hf_to_canonical_map[name]] = tensor_data
    else:
        match = block_re.match(name)
        if match:
            layer_id = int(match.group(1))
            hf_subname = match.group(2)

            canonical_subname = hf_subname
            if hf_subname == "attn.c_attn.weight":
                canonical_subname = "attn_qkv.weight"
            elif hf_subname == "attn.c_attn.bias":
                canonical_subname = "attn_qkv.bias"
            elif hf_subname == "attn.c_proj.weight":
                canonical_subname = "attn_output.weight"
            elif hf_subname == "attn.c_proj.bias":
                canonical_subname = "attn_output.bias"
            elif hf_subname == "ln_1.weight":
                canonical_subname = "attn_norm.weight"
            elif hf_subname == "ln_1.bias":
                canonical_subname = "attn_norm.bias"
            elif hf_subname == "mlp.c_fc.weight":
                canonical_subname = "ffn_up.weight"
            elif hf_subname == "mlp.c_fc.bias":
                canonical_subname = "ffn_up.bias"
            elif hf_subname == "mlp.c_proj.weight":
                canonical_subname = "ffn_down.weight"
            elif hf_subname == "mlp.c_proj.bias":
                canonical_subname = "ffn_down.bias"
            elif hf_subname == "ln_2.weight":
                canonical_subname = "ffn_norm.weight"
            elif hf_subname == "ln_2.bias":
                canonical_subname = "ffn_norm.bias"

            block_key = f"blk.{layer_id}"
            
            full_canonical_name_in_block = f"{block_key}.{canonical_subname}"
            
            # Store all tensors in a single flat "tensors" map with full names.
            export["tensors"][full_canonical_name_in_block] = tensor_data
            
        else:
            # If it's not in the explicit map and not a block tensor, it's "other"
            print(f"⚠️ Warning: Tensor '{name}' (Shape: {tensor.shape}) not mapped to a canonical name and not part of a block. Ensure it's handled or explicitly ignored.")
            if "others" not in export: export["others"] = {} 
            export["others"][name] = tensor_data

# Save the model weights and metadata
weights_fname = os.path.join(args.output_dir, "gpt2_tiny_weights.json") 
with open(weights_fname, "w") as f:
    json.dump(export, f, indent=2)

print(f"✅ Export written to {weights_fname}")