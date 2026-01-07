---
description: Debug a failed GPU CLI run and suggest fixes
---

# GPU Debug

Analyze the error from a failed GPU CLI run and provide actionable fixes.

**Error or context:** $ARGUMENTS

## Instructions

1. **Identify Error Type**
   - OOM (CUDA out of memory)
   - Connection/network issues
   - Model loading failures
   - Sync errors
   - Pod provisioning failures

2. **Collect Information**
   If needed, suggest running:
   - `gpu daemon logs --tail 50`
   - `gpu status`
   - Check gpu.jsonc configuration

3. **Diagnose Root Cause**
   - Match error patterns to known issues
   - Check configuration against requirements
   - Identify mismatches

4. **Provide Fixes**
   - Immediate fix (solve the problem now)
   - Prevention (avoid it in the future)
   - Alternative approaches if primary fix doesn't work

## Output Format

```
## Error Analysis

**Type**: [OOM/Network/Model/Sync/Provisioning]
**Root Cause**: [explanation]

## Immediate Fix

[Step-by-step instructions]

```bash
# Commands to run
```

## Configuration Change (if needed)

```jsonc
{
  // Updated config
}
```

## Prevention

To avoid this in the future:
1. [Tip 1]
2. [Tip 2]
```
