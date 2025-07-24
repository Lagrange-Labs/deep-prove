# gpt2_toy_gguf_export.py
# Requirements: pip install torch gguf transformers

import torch
import torch.nn as nn
import numpy as np
import json
from transformers import GPT2Tokenizer

# ---- Model Definition ----

class ToyGPT2Config:
    def __init__(self, use_real_gpt2_vocab=False):
        if use_real_gpt2_vocab:
            # Load GPT-2 tokenizer to get real vocab size
            from transformers import GPT2Tokenizer
            tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
            self.vocab_size = len(tokenizer.get_vocab())
        else:
            self.vocab_size = 100
        self.n_positions = 32
        self.n_embd = 64
        self.n_layer = 2
        self.n_head = 4

class ToyGPT2(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.n_positions, config.n_embd)
        self.h = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

    def forward(self, idx):
        b, t = idx.size()
        pos = torch.arange(0, t, dtype=torch.long, device=idx.device)
        tok_emb = self.wte(idx)
        pos_emb = self.wpe(pos)[None, :, :]
        x = tok_emb + pos_emb
        for block in self.h:
            x = block(x)
        x = self.ln_f(x)
        logits = self.head(x)
        return logits

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = nn.MultiheadAttention(config.n_embd, config.n_head, batch_first=True)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
        )

    def forward(self, x):
        # Prenorm
        x_norm = self.ln_1(x)
        attn_output, _ = self.attn(x_norm, x_norm, x_norm, need_weights=False)
        x = x + attn_output
        x_norm = self.ln_2(x)
        mlp_output = self.mlp(x_norm)
        x = x + mlp_output
        return x

# ---- Weight Initialization ----

def realistic_init(model):
    for name, param in model.named_parameters():
        if 'weight' in name:
            nn.init.normal_(param, mean=0.0, std=0.02)
        elif 'bias' in name:
            nn.init.zeros_(param)

# ---- Model Statistics ----

def print_model_stats(model, config):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Calculate size in MB (assuming float32, 4 bytes per parameter)
    size_mb = total_params * 4 / (1024 * 1024)
    
    print("=" * 50)
    print("MODEL STATISTICS")
    print("=" * 50)
    print(f"Number of layers: {config.n_layer}")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Approximate size: {size_mb:.2f} MB")
    print(f"Vocabulary size: {config.vocab_size}")
    print(f"Embedding dimension: {config.n_embd}")
    print(f"Number of heads: {config.n_head}")
    print(f"Context length: {config.n_positions}")
    print("=" * 50)

# ---- Tensor Name Mapping ----

def map_pytorch_to_gguf_name(pytorch_name):
    """Map PyTorch parameter names to GGUF tensor names expected by the Rust parser."""
    name_mapping = {
        # Embeddings
        "wte.weight": "token_embd.weight",
        "wpe.weight": "position_embd.weight",
        
        # Final layer norm and projection
        "ln_f.weight": "output_norm.weight", 
        "ln_f.bias": "output_norm.bias",
        "head.weight": "output.weight",
        "head.bias": "output.bias",
    }
    
    # Handle transformer blocks (h.0., h.1., etc. -> blk.0., blk.1., etc.)
    if pytorch_name.startswith("h."):
        # Extract layer number
        parts = pytorch_name.split(".")
        layer_num = parts[1]
        rest = ".".join(parts[2:])
        
        # Map sub-components within each block
        if rest.startswith("ln_1."):
            # Attention layer norm
            param_name = rest[5:]  # Remove "ln_1."
            return f"blk.{layer_num}.attn_norm.{param_name}"
        elif rest.startswith("ln_2."):
            # FFN layer norm  
            param_name = rest[5:]  # Remove "ln_2."
            return f"blk.{layer_num}.ffn_norm.{param_name}"
        elif rest.startswith("attn."):
            # Attention weights - need to handle MultiheadAttention mapping
            attn_part = rest[5:]  # Remove "attn."
            if attn_part.startswith("in_proj_weight"):
                # This is the fused QKV weight from MultiheadAttention
                return f"blk.{layer_num}.attn_qkv.weight"
            elif attn_part.startswith("in_proj_bias"):
                # This is the fused QKV bias from MultiheadAttention
                return f"blk.{layer_num}.attn_qkv.bias"
            elif attn_part.startswith("out_proj.weight"):
                return f"blk.{layer_num}.attn_output.weight"
            elif attn_part.startswith("out_proj.bias"):
                return f"blk.{layer_num}.attn_output.bias"
        elif rest.startswith("mlp."):
            # MLP/FFN weights
            mlp_part = rest[4:]  # Remove "mlp."
            if mlp_part == "0.weight":  # First linear layer
                return f"blk.{layer_num}.ffn_up.weight"
            elif mlp_part == "0.bias":
                return f"blk.{layer_num}.ffn_up.bias"
            elif mlp_part == "2.weight":  # Third layer (after GELU)
                return f"blk.{layer_num}.ffn_down.weight"
            elif mlp_part == "2.bias":
                return f"blk.{layer_num}.ffn_down.bias"
    
    # Check direct mapping
    if pytorch_name in name_mapping:
        return name_mapping[pytorch_name]
    
    # If no mapping found, return original name
    print(f"Warning: No mapping found for parameter '{pytorch_name}', using original name")
    return pytorch_name

# ---- GGUF Export ----

def export_to_gguf(model, config, filename="toy_gpt2.gguf"):
    import gguf
    
    # Create writer with GPT2 architecture
    writer = gguf.GGUFWriter(filename, gguf.MODEL_ARCH_NAMES[gguf.MODEL_ARCH.GPT2])
    
    # Write general metadata
    writer.add_name("toy_gpt2")
    writer.add_description("A toy GPT-2 model for testing")
    writer.add_file_type(1)  # F16
    
    # Write GPT2-specific metadata keys (matching what the Rust parser expects)
    # These are the exact keys the Rust code looks for in zkml/src/parser/gguf.rs
    writer.add_uint32("gpt2.context_length", config.n_positions)
    writer.add_uint32("gpt2.embedding_length", config.n_embd) 
    writer.add_uint32("gpt2.block_count", config.n_layer)
    writer.add_uint32("gpt2.attention.head_count", config.n_head)
    writer.add_float32("gpt2.attention.layer_norm_epsilon", 1e-5)
    
    # Load actual GPT-2 tokenizer
    print("Loading GPT-2 tokenizer...")
    gpt2_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    actual_vocab_size = len(gpt2_tokenizer.get_vocab())
    
    # Add vocab size (use actual GPT-2 vocab size)
    writer.add_vocab_size(actual_vocab_size)
    
    # Extract vocabulary 
    vocab = gpt2_tokenizer.get_vocab()
    print(f"GPT-2 vocab size: {len(vocab)}")
    
    # Create tokens list in order by token ID
    tokens_dict = {v: k for k, v in vocab.items()}
    tokens = []
    for i in range(len(tokens_dict)):
        token_str = tokens_dict[i]
        tokens.append(token_str.encode("utf-8"))
    
    # Extract BPE merges
    merges = []
    
    # Try different methods to get BPE merges
    try:
        # Method 1: Access via backend tokenizer
        if hasattr(gpt2_tokenizer, 'backend_tokenizer'):
            backend = gpt2_tokenizer.backend_tokenizer
            if hasattr(backend, 'get_vocab') and hasattr(backend, 'model'):
                model = backend.model
                if hasattr(model, 'get_vocab'):
                    # This is a more complex approach, let's try a simpler one
                    pass
    except:
        pass
    
    # Method 2: Use the actual merges from the tokenizer's saved files
    try:
        # Download and cache the tokenizer files
        from transformers.utils import cached_file
        merges_file = cached_file("gpt2", "merges.txt")
        if merges_file:
            with open(merges_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                # Skip the header line and process merges
                for line in lines[1:]:  # Skip "#version: 0.2" header
                    line = line.strip()
                    if line:
                        merges.append(line)
    except Exception as e:
        print(f"Could not load merges.txt: {e}")
    
    # Method 3: Fallback - extract from tokenizer internals if available
    if not merges:
        try:
            # Try accessing through different internal attributes
            if hasattr(gpt2_tokenizer, '_tokenizer'):
                tokenizer_obj = gpt2_tokenizer._tokenizer
                if hasattr(tokenizer_obj, 'get_vocab'):
                    # This approach might work for some versions
                    pass
        except:
            pass
    
    # Method 4: Final fallback - create essential BPE merges manually
    if not merges:
        print("Warning: Could not load BPE merges, using fallback set")
        # Essential GPT-2 BPE merges for basic functionality
        merges = [
            "Ġ t", "h e", "i n", "r e", "Ġ a", "e r", "Ġ s", "o n", "a t", "e n",
            "Ġ w", "o r", "i t", "Ġ c", "i s", "e s", "a n", "a l", "Ġ b", "n d",
            "o u", "Ġ f", "o w", "Ġth e", "l e", "i ng", "Ġ m", "a r", "Ġ p",
            "o m", "a s", "o l", "i on", "Ġ h", "u r", "c h", "l l", "en t",
            "Ġ d", "Ġ n", "u s", "a y", "c e", "f t", "Ġ g", "p t", "Ġ l",
            "t h", "s t", "a d", "Ġ o", "r o", "Ġ r", "u n", "e t", "a c",
            "o g", "e d", "u t", "n t", "x t", "Ġ v", "i d", "o s", "r i",
            "a g", "v e", "l y", "Ġs k", "s ky", "Ġsky", "Ġi s", "Ġis"
        ]
    
    print(f"Loaded {len(merges)} BPE merges")
    
    # Verify vocab size matches
    assert len(tokens) == config.vocab_size, f"Token count {len(tokens)} doesn't match config vocab size {config.vocab_size}"
    print(f"Vocab size verified: {config.vocab_size}")
    
    scores = [0.0] * len(tokens)
    token_types = [1] * len(tokens)  # Normal tokens
    
    # Find special token IDs from the actual tokenizer
    eos_token_id = gpt2_tokenizer.eos_token_id if gpt2_tokenizer.eos_token_id is not None else 50256
    bos_token_id = gpt2_tokenizer.bos_token_id if gpt2_tokenizer.bos_token_id is not None else eos_token_id  # GPT-2 uses EOS as BOS
    unk_token_id = gpt2_tokenizer.unk_token_id if gpt2_tokenizer.unk_token_id is not None else eos_token_id
    pad_token_id = gpt2_tokenizer.pad_token_id if gpt2_tokenizer.pad_token_id is not None else eos_token_id
    
    # Set token types for special tokens
    if eos_token_id < len(token_types):
        token_types[eos_token_id] = 3  # Control token
    if bos_token_id < len(token_types) and bos_token_id != eos_token_id:
        token_types[bos_token_id] = 3  # Control token
    if unk_token_id < len(token_types) and unk_token_id not in [eos_token_id, bos_token_id]:
        token_types[unk_token_id] = 2  # Unknown token
    
    print(f"Special tokens - EOS: {eos_token_id}, BOS: {bos_token_id}, UNK: {unk_token_id}, PAD: {pad_token_id}")
    
    # Add tokenizer metadata
    writer.add_tokenizer_model("gpt2")
    writer.add_token_list(tokens)
    writer.add_token_scores(scores)
    writer.add_token_types(token_types)
    writer.add_token_merges(merges)
    
    # Add special token IDs using actual GPT-2 values
    writer.add_bos_token_id(bos_token_id)
    writer.add_eos_token_id(eos_token_id)
    writer.add_unk_token_id(unk_token_id)
    writer.add_pad_token_id(pad_token_id)
    
    print(f"Added {len(tokens)} tokens and {len(merges)} merges to tokenizer")
    
    # Write weights with proper name mapping
    print("\nMapping tensor names:")
    for pytorch_name, param in model.named_parameters():
        gguf_name = map_pytorch_to_gguf_name(pytorch_name)
        print(f"  {pytorch_name} -> {gguf_name}")
        writer.add_tensor(gguf_name, param.detach().cpu().numpy())
    
    # Write to file
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

# ---- Main ----

if __name__ == "__main__":
    # Use real GPT-2 vocab size to match the tokenizer
    config = ToyGPT2Config(use_real_gpt2_vocab=True)
    model = ToyGPT2(config)
    realistic_init(model)
    
    # Print model statistics
    print_model_stats(model, config)
    
    export_to_gguf(model, config)
    print("Exported toy GPT-2 model to toy_gpt2.gguf")
