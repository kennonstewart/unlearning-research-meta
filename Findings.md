# ρ Propagation Audit — Findings

## Call Flow (event → accountant → model metrics → event row → seed summary)
- [memory_pair.py:479] delete(...) -> calls _delete_with_accountant(x, y)
- [memory_pair.py:484] _delete_with_accountant(...) -> accountant.pre_delete(sensitivity) 
- [memory_pair.py:526] **EXPECTED: accountant.spend(sensitivity, sigma)** ✗ **BLOCKED by regret_gate**
- [memory_pair.py:689] get_metrics_dict() -> returns accountant.metrics() containing rho_spent
- [phases.py:183] _create_extended_log_entry(...) merges model_metrics (force-update confirmed working)
- [grid_runner.py] process_seed_output(...) reads last rho column from events

## Root Cause Discovery

**DEBUG OUTPUT EVIDENCE:**
```
[DEBUG] regret gate check: proj_avg=6733044023.955882, gamma_delete=0.45000000000000007
[DEBUG] regret gate blocked deletion
```

**All delete operations are blocked by the regret gate before accountant.spend() is called.**

## Hypothesis Analysis

### Hypothesis 1: Accountant never called per event ✓ **CONFIRMED**
- **Evidence**: Debug output shows `_delete_with_accountant` is called for every delete
- **Evidence**: `accountant.pre_delete()` succeeds and returns valid sigma values  
- **Evidence**: Regret gate blocks ALL deletions with `proj_avg >> gamma_delete`
- **Evidence**: `accountant.spend()` debug output never appears
- **Conclusion**: **ROOT CAUSE - Regret gate prevents accountant.spend() from being called**

### Hypothesis 2: Per-op noise computed but not accounted ✓ **RELATED** 
- **Evidence**: `sigma_delete = 1.996834` appears in logs (noise computed correctly)
- **Evidence**: `rho_spent = 0.0` throughout (never incremented)
- **Conclusion**: Noise is computed by `pre_delete()` but spend() never called due to regret gate

### Hypothesis 3: Odometer reset/scope issue ✗ **FALSE**
- **Evidence**: Debug shows odometer logic never reached
- **Conclusion**: Not applicable - spend() never called

### Hypothesis 4: Column naming/overwrite ✗ **FALSE**
- **Evidence**: Privacy metrics force-update works correctly in tests
- **Evidence**: Columns `rho_spent`, `privacy_spend_running` exist and propagate properly
- **Conclusion**: Data pipeline is sound

### Hypothesis 5: Seed aggregator picks wrong source ✗ **FALSE**
- **Evidence**: Aggregation works correctly in tests with non-zero rho values
- **Conclusion**: Not the issue - source problem is that rho_spent never increments

### Hypothesis 6: Config disables accounting ✗ **FALSE**
- **Evidence**: `accountant = "zcdp"`, `privacy_mode = True`, `log_privacy_spend = True`
- **Evidence**: Accountant is initialized and `pre_delete()` works
- **Conclusion**: Accounting is enabled but blocked by regret constraints

## Technical Details

**Regret Gate Logic (memory_pair.py:499-524):**
```python
if hasattr(self.cfg, "gamma_delete") and self.cfg.gamma_delete is not None:
    proj_avg = (ins_reg + del_reg) / max(self.events_seen or 1, 1)
    if proj_avg > getattr(self.cfg, "gamma_delete", float("inf")):
        return "regret_gate"  # ← ALL DELETES BLOCKED HERE
```

**Observed Values:**
- `proj_avg`: ~6.7 billion (theoretical regret projection)
- `gamma_delete`: 0.45 (regret budget allocation)
- **Result**: `6.7e9 >> 0.45` → regret gate blocks all deletions

## Surgical Fix Plan (minimal)

**Option A: Disable Regret Gate (Simplest)**
- **Patch point**: `memory_pair.py:523` - Add condition to bypass regret gate for debugging
- **Change**: `if proj_avg > gamma_delete and not getattr(self.cfg, "disable_regret_gate", False):`
- **Config fix**: Set `config.disable_regret_gate = True` in debug scenarios

**Option B: Fix Regret Budget (Proper)**  
- **Root cause**: `gamma_delete = 0.45` is too small for the theoretical regret bounds
- **Patch point**: `config.py` or experiment setup - increase regret budget  
- **Change**: Use `gamma_delete = 10.0` or higher, or fix regret bound calculation

**Option C: Log Blocked Deletions Properly**
- **Patch point**: `phases.py` - ensure blocked deletions don't appear as successful deletes in logs
- **Change**: Add `blocked_reason` column to distinguish actual vs blocked deletions

**RECOMMENDED**: Option A for immediate debugging + Option C for proper logging

## Verification Tests
1. **Regret gate bypass test**: Set `disable_regret_gate=True`, verify rho increments
2. **Blocked deletion logging**: Verify `blocked_reason="regret_gate"` appears in logs  
3. **Privacy accounting test**: Confirm `rho_spent` increments with successful deletions

## Quick Reproducer (pre/post fix)

**Pre-fix (current behavior):**
```bash
python debug_rho_probe.py
# Expected: rho_spent stays 0.0, all deletions blocked by regret_gate
```

**Post-fix (with regret gate disabled):**
```bash  
python debug_rho_probe.py  # (with disable_regret_gate=True)
# Expected: rho_spent increments from 0 → ~X over Y successful deletes
```

## Key Discovery

**The privacy accounting code is CORRECT** - the issue is that the regret gate prevents any deletions from actually proceeding to the privacy spending step. This is a **configuration/constraint** issue, not a privacy accounting bug.