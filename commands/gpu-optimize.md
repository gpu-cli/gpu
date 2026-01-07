---
description: Analyze and optimize GPU CLI configuration for cost and performance
---

# GPU Optimize

Analyze the current project's GPU CLI configuration and suggest optimizations.

## Instructions

1. **Read Current Configuration**
   - Look for `gpu.jsonc` or `gpu.toml` in the current directory
   - If not found, ask user about their workload

2. **Analyze GPU Selection**
   - Is the GPU over-provisioned? (paying for unused VRAM)
   - Is the GPU under-provisioned? (risking OOM errors)
   - Are there cheaper alternatives that meet requirements?

3. **Check Configuration**
   - Are outputs configured correctly?
   - Is cooldown_minutes appropriate for the use case?
   - Are model downloads pre-configured?

4. **Provide Recommendations**
   - Specific config changes with reasoning
   - Cost comparison (current vs optimized)
   - Trade-offs explained

## Output Format

```
## Current Configuration Analysis

**GPU**: [current GPU] @ $X.XX/hr
**VRAM Utilization**: [estimated %]
**Issue**: [over/under provisioned or optimal]

## Recommendations

### 1. [Recommendation Title]
**Change**: [specific change]
**Savings**: $X.XX/hr (X%)
**Trade-off**: [any downsides]

### Optimized Configuration

```jsonc
{
  // Updated config
}
```

**Estimated Monthly Savings**: $XXX (assuming Y hours/month)
```
