import torch
import re
import json
import argparse
import os
from transformers import GPT2Model, GPT2Tokenizer
from transformers import AutoModelForCausalLM
from transformers.modeling_utils import Conv1D

torch.set_printoptions(precision=6, sci_mode=False)

# --- Argument Parsing --- #
parser = argparse.ArgumentParser(description="Run a GPT-2 model and dump weights and intermediate tensors.")
parser.add_argument(
    "--output-dir",
    type=str,
    default=".",
    help="Directory to save the output JSON files."
)
parser.add_argument(
    "--model-name",
    type=str,
    default="tiny-gpt2",
    help="The Hugging Face model ID to use (e.g., 'distilgpt2','tiny-gpt2')."
)
parser.add_argument(
    '--export-model',
    action='store_true',
    help="If set, exports the model weights to a JSON file."
)
args = parser.parse_args()

# --- Model Selection ---
model_map = {
    "tiny-gpt2": "sshleifer/tiny-gpt2",
    "distilgpt2": "distilbert/distilgpt2",
}
model_hf_id = model_map.get(args.model_name, args.model_name)  # Use the mapping or the direct name
print(f"ℹ️ Using model: {model_hf_id}")

# Ensure output directory exists
if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)
    print(f"Created output directory: {args.output_dir}")

# Load tokenizer and model
print(f"ℹ️ Loading model: {model_hf_id}")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained(model_hf_id)

model.eval()
# --- Architectural Investigation ---
print("--- Model Config ---")
print(model.config)
print("\n--- Model Structure ---")
print(model)
print("\n" + "="*50 + "\n")
# --- End Architectural Investigation ---
# Input
input_text = "The dog is cute"
input_ids = tokenizer.encode(input_text, return_tensors="pt")

# Run through model to get reference
with torch.no_grad():
    # IMPORTANT: Call the transformer directly to get clean hidden states
    transformer_outputs = model.transformer(input_ids, output_hidden_states=True)
    inputs_embeds = transformer_outputs.hidden_states[0] 

    # Explicitly calculate token and positional embeddings
    input_seq_length = input_ids.size(1)
    position_ids = torch.arange(0, input_seq_length, dtype=torch.long, device=input_ids.device).unsqueeze(0)
    
    token_embeds_only = model.transformer.wte(input_ids)
    pos_embeds_only = model.transformer.wpe(position_ids)

    # Sanity check: their sum should be the inputs_embeds
    assert torch.allclose(inputs_embeds, token_embeds_only + pos_embeds_only, atol=1e-6), \
        "Sum of explicit token and positional embeddings does not match hidden_states[0]"

# ---- Manual forward through all transformer blocks ---- #
def to_list(t): return t.squeeze().flatten().tolist()

layer_debug_outputs = []
current_hidden_state = inputs_embeds

with torch.no_grad():
    def split_heads(x):
        B, T, C = x.size()
        H = model.config.n_head
        return x.view(B, T, H, C // H).transpose(1, 2)

    def merge_heads(x):
        B, H, T, D = x.size()
        return x.transpose(1, 2).contiguous().view(B, T, H * D)

    for i, layer in enumerate(model.transformer.h):
        print(f"--- Processing Layer {i} ---")
        
        hidden_states = current_hidden_state

        # 1. LayerNorm 1
        ln1_out = layer.ln_1(hidden_states)
        
        # Debug: Check if we're using the reference intermediate from layer 5
        if i == 5:
            reference_ln1_out = layer.ln_1(transformer_outputs.hidden_states[i])
            ln1_diff = (ln1_out - reference_ln1_out).abs().max()
            print(f"🔍 Layer {i} ln1_out vs reference: max diff = {ln1_diff.item():.8f}")
            if ln1_diff > 1e-6:
                print(f"❌ Layer {i} ln1_out already diverged from reference!")
                print(f"  Manual ln1_out first 3: {ln1_out.flatten()[:3].tolist()}")
                print(f"  Reference ln1_out first 3: {reference_ln1_out.flatten()[:3].tolist()}")
        
        # 2. Attention Block
        # To capture intermediates, we call the sub-modules of the attention layer
        
        # a. QKV projection
        q_before_heads, k_before_heads, v_before_heads = layer.attn.c_attn(ln1_out).split(model.config.hidden_size, dim=2)
        
        # b. Split heads for multi-head attention
        q = split_heads(q_before_heads)
        k = split_heads(k_before_heads)
        v = split_heads(v_before_heads)
        
        # c. Core attention mechanism
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (q.size(-1) ** 0.5)
        mask = torch.tril(torch.ones_like(attn_scores[0, 0])).unsqueeze(0).unsqueeze(0)
        attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights_after_dropout = layer.attn.attn_dropout(attn_weights)
        attn_output_merged = torch.matmul(attn_weights_after_dropout, v)
        attn_output = merge_heads(attn_output_merged)
        
        # d. Final projection
        attn_output_proj = layer.attn.c_proj(attn_output)
        attn_output_proj = layer.attn.resid_dropout(attn_output_proj)
        
        # 3. First residual connection
        residual_attn = hidden_states + attn_output_proj
        
        # 4. LayerNorm 2
        ln2_out = layer.ln_2(residual_attn)
        
        # 5. MLP block
        # To capture intermediates, we call the sub-modules of the MLP
        ffn_intermediate = layer.mlp.c_fc(ln2_out)
        ffn_activated = layer.mlp.act(ffn_intermediate)
        ffn_output = layer.mlp.c_proj(ffn_activated)
        ffn_output = layer.mlp.dropout(ffn_output)
        
        # 6. Second residual connection
        manual_output = residual_attn + ffn_output
        
        # 7. Apply final LayerNorm if this is the last layer
        manual_output_with_final_ln = None
        if i == len(model.transformer.h) - 1:  # Last layer
            manual_output_with_final_ln = model.transformer.ln_f(manual_output)
        
        # Final validation against the transformer's output
        automated_output = transformer_outputs.hidden_states[i+1]
        
        # Choose the correct manual output for comparison based on layer position
        if i == len(model.transformer.h) - 1:  # Last layer
            # For the final layer, compare against the output with final LayerNorm applied
            comparison_manual_output = manual_output_with_final_ln
        else:
            # For intermediate layers, compare against the raw layer output
            comparison_manual_output = manual_output
        
        max_diff = (comparison_manual_output - automated_output).abs().max()
        
        if not torch.allclose(comparison_manual_output, automated_output, atol=1e-5):
            print(f"❌ Mismatch in Layer {i}. Max diff: {max_diff.item():.6f}")
            
            # For layer 5, let's debug step by step
            if i == 5:
                print(f"🔍 Step-by-step debug for Layer {i}:")
                ref_hidden_states = transformer_outputs.hidden_states[i]
                ref_ln1_out = layer.ln_1(ref_hidden_states)
                print(f"  Hidden states match: {torch.allclose(hidden_states, ref_hidden_states, atol=1e-6)}")
                print(f"  LN1 outputs match: {torch.allclose(ln1_out, ref_ln1_out, atol=1e-6)}")
                
                # Check attention calculation
                ref_attn_out = layer.attn(ref_ln1_out)[0]
                our_attn_out = attn_output_proj
                print(f"  Attention outputs match: {torch.allclose(our_attn_out, ref_attn_out, atol=1e-6)}")
                
                # Check residual after attention
                ref_residual_attn = ref_hidden_states + ref_attn_out
                print(f"  Residual attn match: {torch.allclose(residual_attn, ref_residual_attn, atol=1e-6)}")
                
                # Check MLP
                ref_ln2_out = layer.ln_2(ref_residual_attn)
                ref_mlp_out = layer.mlp(ref_ln2_out)
                our_mlp_out = ffn_output
                print(f"  LN2 outputs match: {torch.allclose(ln2_out, ref_ln2_out, atol=1e-6)}")
                print(f"  MLP outputs match: {torch.allclose(our_mlp_out, ref_mlp_out, atol=1e-6)}")
                
                # Check final sum
                ref_final = ref_residual_attn + ref_mlp_out
                our_final = residual_attn + ffn_output
                print(f"  Final outputs match: {torch.allclose(our_final, ref_final, atol=1e-6)}")
                print(f"  Reference final (first 3): {ref_final.flatten()[:3].tolist()}")
                print(f"  Our final (first 3): {our_final.flatten()[:3].tolist()}")
                print(f"  Manual calc (first 3): {manual_output.flatten()[:3].tolist()}")
                print(f"  Automated (first 3): {automated_output.flatten()[:3].tolist()}")
                
                # Sanity check: manual_output should equal our_final
                print(f"  manual_output == our_final: {torch.allclose(manual_output, our_final, atol=1e-8)}")
                
                # Double check the transformer reference calculation
                ref_block_output = layer(ref_hidden_states)[0]
                print(f"  Direct block call matches automated: {torch.allclose(ref_block_output, automated_output, atol=1e-6)}")
                
                # Test: What if we call the SAME layer on the SAME hidden state multiple times?
                ref_block_output2 = layer(ref_hidden_states)[0]
                print(f"  Layer is deterministic: {torch.allclose(ref_block_output, ref_block_output2, atol=1e-8)}")
                
                # The issue might be that transformer_outputs.hidden_states comes from a different execution context
                print(f"  Block output diff from transformer: {(ref_block_output - automated_output).abs().max().item():.8f}")
                print(f"  Block vs transformer (first 3): block={ref_block_output.flatten()[:3].tolist()}")
                print(f"  Block vs transformer (first 3): transformer={automated_output.flatten()[:3].tolist()}")
        else:
            print(f"✅ Layer {i} output matches. Max diff: {max_diff.item():.6f}")
        
        # Dump intermediate tensors for this layer
        layer_json = {
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
            "ffn_output_proj": to_list(ffn_output),
            "manual_output": to_list(manual_output),
            "automated_output": to_list(automated_output),
            "ffn_after_gelu": to_list(ffn_activated),
            "ffn_after_down": to_list(ffn_output),
            "ffn_after_add": to_list(manual_output)
        }
        
        # Add final LayerNorm output if this is the last layer
        if manual_output_with_final_ln is not None:
            layer_json["manual_output_with_final_ln"] = to_list(manual_output_with_final_ln)
        
        layer_debug_outputs.append(layer_json)

        # Update hidden state for the next loop iteration
        # CRITICAL: Use the correct automated output, not the manual output
        # This prevents error accumulation across layers
        current_hidden_state = automated_output

# Dump
output_json = {
    "token": input_text,
    "input_ids": input_ids.squeeze().tolist(),
    "token_embeds_only": to_list(token_embeds_only),
    "pos_embeds_only": to_list(pos_embeds_only),
    "inputs_embeds": to_list(inputs_embeds),
    "layers": layer_debug_outputs,
    "final_output": to_list(manual_output_with_final_ln) if manual_output_with_final_ln is not None else None,
}

# Get the final token prediction using argmax
with torch.no_grad():
    # Get our manual prediction
    manual_logits = model.lm_head(manual_output_with_final_ln)
    
    # Get official model's final projection
    official_outputs = model(input_ids)
    official_logits = official_outputs.logits
    
    # Compare the final projections
    logits_diff = (manual_logits - official_logits).abs().max()
    print(f"🔍 Final projection comparison:")
    print(f"  Max difference in logits: {logits_diff.item():.6f}")
    print(f"  Logits match: {torch.allclose(manual_logits, official_logits, atol=1e-5)}")
    
    # Use argmax for debug output
    next_token = torch.argmax(manual_logits[:, -1, :], dim=-1)
    
    output_json["next_token_id"] = next_token.item()
    output_json["logits"] = to_list(manual_logits)
    print(f"LOGITS: {output_json['logits'][0:5]}")
    output_json["logits_max_diff"] = logits_diff.item()

# Save the debug tensor outputs
output_debug_fname = os.path.join(args.output_dir, f"{args.model_name.replace('-', '_')}_debug_output.json")
with open(output_debug_fname, "w") as f:
    json.dump(output_json, f, indent=2)
# Print absolute path for clarity
print(f"✅ Debug outputs written to {os.path.abspath(output_debug_fname)}")

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

                    current_data_list = W_q_orig.flatten().tolist() + \
                                        W_k_orig.flatten().tolist() + \
                                        W_v_orig.flatten().tolist()
                    # Shape for JSON is 1D concatenated data
                    current_shape = [len(current_data_list)]
                    print(f"ℹ️ Exporting {name} : Original Conv1D shape {list(processed_tensor.shape)}. Exporting concatenated W_q.T, W_k.T, W_v.T data as 1D array, shape {current_shape}.")
                elif hf_subname == "attn.c_proj.weight":
                     current_shape = list(processed_tensor.shape)
                     current_data_list = processed_tensor.numpy().flatten().tolist()
                     print(f"ℹ️ Exporting {name}: Original Conv1D shape {list(processed_tensor.shape)}, NOT transposing.")
                     print(f"\t\t{current_data_list}\n")
                elif hf_subname.endswith(".weight"): # Other Conv1D weights (e.g., mlp.c_fc, mlp.c_proj, attn.c_proj)
                    # These are [in_features, out_features]. Transpose to [out_features, in_features].
                    transposed_tensor = processed_tensor
                    current_shape = list(transposed_tensor.shape)
                    current_data_list = transposed_tensor.numpy().flatten().tolist()
                    print(f"ℹ️ Exporting {name}: Original Conv1D shape {list(processed_tensor.shape)}, transposing to {current_shape} for export.")
                    print(f"\t\t{current_data_list}\n")
                
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

# Save the model weights and metadata if requested
if args.export_model:
    weights_fname = os.path.join(args.output_dir, f"{args.model_name.replace('-', '_')}_weights.json") 
    with open(weights_fname, "w") as f:
        json.dump(export, f, indent=2)
    # Print absolute path for clarity
    print(f"✅ Export written to {os.path.abspath(weights_fname)}")