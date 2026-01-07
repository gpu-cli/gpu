---
description: Create a complete GPU CLI project from a natural language description
---

# GPU Create

Transform the user's request into a complete, runnable GPU CLI project.

**User's request:** $ARGUMENTS

## Instructions

1. **Understand the Intent**
   - What type of task? (training, inference, processing, generation)
   - What domain? (images, text/LLM, audio, video)
   - What scale? (one-off, batch, API service)

2. **Generate Complete Project**
   Create all necessary files:
   - `gpu.jsonc` - GPU CLI configuration with optimal GPU selection
   - Main script (Python) - Complete, working implementation
   - `requirements.txt` - All dependencies
   - `README.md` - Usage instructions

3. **Select Optimal GPU**
   Use the gpu-cost-optimizer skill to choose the right GPU based on:
   - Model VRAM requirements
   - Cost efficiency
   - Availability

4. **Provide Cost Estimate**
   Always include estimated costs for the workflow.

5. **Write Files to Current Directory**
   Use the Write tool to create all files in the user's current working directory.

## Quality Standards

- Every project must work with `gpu run` immediately
- Include error handling and progress tracking
- Document all configuration options
- Estimate costs transparently
