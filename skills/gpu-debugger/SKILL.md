---
name: gpu-debugger
description: "Debug failed GPU CLI runs. Analyze error messages, diagnose OOM errors, fix sync issues, troubleshoot connectivity, and resolve common problems. Turn cryptic errors into actionable fixes."
---

# GPU Debugger

**Turn errors into solutions.**

This skill helps debug failed GPU CLI runs: OOM errors, sync failures, connectivity issues, model loading problems, and more.

## When to Use This Skill

| Problem | This Skill Helps With |
|---------|----------------------|
| "CUDA out of memory" | OOM diagnosis and fixes |
| "Connection refused" | Connectivity troubleshooting |
| "Sync failed" | File sync debugging |
| "Pod won't start" | Provisioning issues |
| "Model won't load" | Model loading errors |
| "Command exited with error" | Exit code analysis |
| "My run is hanging" | Stuck process diagnosis |

## Debugging Workflow

```
Error occurs
     │
     ▼
┌─────────────────────────┐
│ 1. Collect information  │
│    - Error message      │
│    - Daemon logs        │
│    - Exit code          │
│    - VRAM usage         │
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│ 2. Identify error type  │
│    - OOM                │
│    - Network            │
│    - Model              │
│    - Sync               │
│    - Permission         │
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│ 3. Apply fix            │
│    - Config change      │
│    - Code change        │
│    - Retry              │
└─────────────────────────┘
```

## Information Collection Commands

### Check Daemon Logs

```bash
# Last 50 log lines
gpu daemon logs --tail 50

# Full logs since last restart
gpu daemon logs

# Follow logs in real-time
gpu daemon logs --follow
```

### Check Pod Status

```bash
# Current pod status
gpu status

# Pod details
gpu pods list
```

### Check Job History

```bash
# Recent jobs
gpu jobs list

# Specific job details
gpu jobs show <job-id>
```

## Common Errors and Solutions

### 1. CUDA Out of Memory (OOM)

**Error messages:**
```
CUDA out of memory. Tried to allocate X GiB
RuntimeError: CUDA error: out of memory
torch.cuda.OutOfMemoryError
```

**Diagnosis:**
```bash
# In your script, check VRAM usage
import torch
print(f"VRAM allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
print(f"VRAM reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
```

**Solutions by severity:**

| Solution | VRAM Savings | Effort |
|----------|-------------|--------|
| Reduce batch size | ~Linear | Easy |
| Enable gradient checkpointing | ~40% | Easy |
| Use FP16/BF16 | ~50% | Easy |
| Use INT8 quantization | ~50% | Medium |
| Use INT4 quantization | ~75% | Medium |
| Enable CPU offloading | Variable | Easy |
| Use larger GPU | Solves it | $$ |

**Quick fixes:**

```python
# Reduce batch size
BATCH_SIZE = 1  # Start small, increase until OOM

# Enable gradient checkpointing
model.gradient_checkpointing_enable()

# Use FP16
model = model.half()

# Use INT4 quantization
from transformers import BitsAndBytesConfig
bnb_config = BitsAndBytesConfig(load_in_4bit=True)
model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config)

# CPU offloading (for diffusers)
pipe.enable_model_cpu_offload()

# Clear cache between batches
torch.cuda.empty_cache()
```

**Config fix:**
```jsonc
{
  // Upgrade to larger GPU
  "gpu_type": "A100 PCIe 80GB",  // Instead of RTX 4090
  "min_vram": 80
}
```

### 2. Connection Refused / Timeout

**Error messages:**
```
Connection refused
Connection timed out
SSH connection failed
Failed to connect to daemon
```

**Diagnosis:**
```bash
# Check daemon status
gpu daemon status

# Check if daemon is running
ps aux | grep gpud

# Check daemon logs
gpu daemon logs --tail 20
```

**Solutions:**

| Cause | Solution |
|-------|----------|
| Daemon not running | `gpu daemon start` |
| Daemon crashed | `gpu daemon restart` |
| Wrong socket | Check `GPU_DAEMON_SOCKET` env var |
| Port conflict | Kill conflicting process |

**Restart daemon:**
```bash
gpu daemon stop
gpu daemon start
```

### 3. Pod Won't Start / Provisioning Failed

**Error messages:**
```
Failed to create pod
No GPUs available
Insufficient resources
Provisioning timeout
```

**Diagnosis:**
```bash
# Check available GPUs
gpu machines list

# Check specific GPU availability
gpu machines list --gpu "RTX 4090"
```

**Solutions:**

| Cause | Solution |
|-------|----------|
| GPU type unavailable | Try different GPU type |
| Region full | Remove region constraint |
| Price too low | Increase max_price |
| Volume region mismatch | Use volume's region |

**Config fixes:**
```jsonc
{
  // Use min_vram instead of exact GPU
  "gpu_type": null,
  "min_vram": 24,  // Any GPU with 24GB+

  // Or try different GPU
  "gpu_type": "RTX A6000",  // Alternative to RTX 4090

  // Or relax region constraint
  "region": null,  // Any region

  // Or increase price tolerance
  "max_price": 2.0  // Allow up to $2/hr
}
```

### 4. Sync Errors

**Error messages:**
```
rsync error
Sync failed
File not found
Permission denied during sync
```

**Diagnosis:**
```bash
# Check sync status
gpu sync status

# Check .gitignore
cat .gitignore

# Check outputs config
cat gpu.jsonc | grep outputs
```

**Solutions:**

| Cause | Solution |
|-------|----------|
| File too large | Add to .gitignore |
| Permission issue | Check file permissions |
| Path not in outputs | Add to outputs config |
| Disk full on pod | Increase workspace size |

**Config fixes:**
```jsonc
{
  // Ensure outputs are configured
  "outputs": ["output/", "results/", "models/"],

  // Exclude large files
  "exclude_outputs": ["*.tmp", "*.log", "checkpoints/"],

  // Increase storage
  "workspace_size_gb": 100
}
```

### 5. Model Loading Errors

**Error messages:**
```
Model not found
Could not load model
Safetensors error
HuggingFace rate limit
```

**Diagnosis:**
```bash
# Check if model is downloading
# Look for download progress in job output

# Check HuggingFace cache on pod
gpu run ls -la ~/.cache/huggingface/hub/
```

**Solutions:**

| Cause | Solution |
|-------|----------|
| Model not downloaded | Add to download spec |
| Wrong model path | Fix path in code |
| HF rate limit | Set HF_TOKEN |
| Network issue | Retry with timeout |
| Gated model | Accept license on HF |

**Config fixes:**
```jsonc
{
  // Pre-download models
  "download": [
    { "strategy": "hf", "source": "meta-llama/Llama-3.1-8B-Instruct", "timeout": 7200 }
  ],

  // Set HF token in environment
  "environment": {
    "shell": {
      "steps": [
        { "run": "echo 'HF_TOKEN=your_token' >> ~/.bashrc" }
      ]
    }
  }
}
```

### 6. Process Hanging / Stuck

**Symptoms:**
- No output for a long time
- Process doesn't exit
- `gpu status` shows job running forever

**Diagnosis:**
```bash
# Check if process is actually running
gpu run ps aux | grep python

# Check for infinite loops in logs
gpu jobs logs <job-id> --tail 100

# Check VRAM (might be swapping)
gpu run nvidia-smi
```

**Solutions:**

| Cause | Solution |
|-------|----------|
| Infinite loop | Fix code logic |
| Waiting for input | Make script non-interactive |
| VRAM thrashing | Reduce memory usage |
| Deadlock | Add timeout |
| Network wait | Add timeout to requests |

**Code fixes:**
```python
# Add timeout to model loading
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    low_cpu_mem_usage=True  # Prevent OOM during loading
)

# Add timeout to HTTP requests
import requests
response = requests.get(url, timeout=30)

# Add progress bars to see activity
from tqdm import tqdm
for item in tqdm(items):
    process(item)
```

### 7. Exit Code Errors

**Common exit codes:**

| Code | Meaning | Common Cause |
|------|---------|--------------|
| 0 | Success | - |
| 1 | General error | Script exception |
| 2 | Misuse of command | Bad arguments |
| 126 | Permission denied | Script not executable |
| 127 | Command not found | Missing binary |
| 137 | Killed (OOM) | Out of memory |
| 139 | Segfault | Bad memory access |
| 143 | Terminated | Killed by signal |

**Diagnosis:**
```bash
# Check last job exit code
gpu jobs list --limit 1

# Get full output including error
gpu jobs logs <job-id>
```

**Solutions:**

| Exit Code | Solution |
|-----------|----------|
| 1 | Check Python traceback, fix exception |
| 126 | `chmod +x script.sh` |
| 127 | Install missing package |
| 137 | Reduce memory usage, bigger GPU |
| 139 | Update PyTorch/CUDA versions |

### 8. CUDA Version Mismatch

**Error messages:**
```
CUDA error: no kernel image is available
CUDA version mismatch
Torch not compiled with CUDA enabled
```

**Diagnosis:**
```bash
# Check CUDA version on pod
gpu run nvcc --version
gpu run nvidia-smi | head -3

# Check PyTorch CUDA version
gpu run python -c "import torch; print(torch.version.cuda)"
```

**Solutions:**
```jsonc
{
  // Use a known-good base image
  "environment": {
    "base_image": "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"
  }
}
```

Or in requirements.txt:
```
# Install PyTorch with specific CUDA version
--extra-index-url https://download.pytorch.org/whl/cu124
torch==2.4.0
```

## Debug Script Template

Add this to your projects for better error info:

```python
#!/usr/bin/env python3
"""Wrapper script with debugging info."""

import sys
import traceback
import torch

def print_system_info():
    """Print system info for debugging."""
    print("=" * 50)
    print("SYSTEM INFO")
    print("=" * 50)
    print(f"Python: {sys.version}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM total: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        print(f"VRAM allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    print("=" * 50)

def main():
    # Your actual code here
    pass

if __name__ == "__main__":
    print_system_info()
    try:
        main()
    except Exception as e:
        print("\n" + "=" * 50)
        print("ERROR OCCURRED")
        print("=" * 50)
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {str(e)}")
        print("\nFull traceback:")
        traceback.print_exc()

        # Print memory info for OOM debugging
        if torch.cuda.is_available():
            print(f"\nVRAM at error: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

        sys.exit(1)
```

## Quick Reference: Error → Solution

| Error Contains | Likely Cause | Quick Fix |
|----------------|--------------|-----------|
| `CUDA out of memory` | OOM | Reduce batch size, use quantization |
| `Connection refused` | Daemon down | `gpu daemon restart` |
| `No GPUs available` | Supply shortage | Try different GPU type |
| `Model not found` | Not downloaded | Add to download spec |
| `Permission denied` | File permissions | `chmod +x` or check path |
| `Killed` | OOM (exit 137) | Bigger GPU |
| `Timeout` | Network/hanging | Add timeout, check code |
| `CUDA version` | Version mismatch | Use compatible base image |
| `rsync error` | Sync issue | Check .gitignore, outputs |
| `rate limit` | HuggingFace limit | Set HF_TOKEN |

## Output Format

When debugging:

```markdown
## Debug Analysis

### Error Identified

**Type**: [OOM/Network/Model/Sync/etc.]
**Message**: `[exact error message]`

### Root Cause

[Explanation of why this happened]

### Solution

**Option 1** (Recommended): [solution]
```[code/config change]```

**Option 2** (Alternative): [solution]
```[code/config change]```

### Prevention

To avoid this in the future:
1. [Prevention tip 1]
2. [Prevention tip 2]
```
