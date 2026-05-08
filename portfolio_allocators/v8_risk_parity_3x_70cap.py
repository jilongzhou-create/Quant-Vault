import math
from portfolio_allocators.base_allocator import BaseAllocator
from portfolio_allocators import register_allocator


@register_allocator
class V8RiskParity3x70Cap(BaseAllocator):
    name = "V8RiskParity3x70Cap"
    description = "V4 Risk Parity + 3x overall leverage + 70% single-asset cap (Pro-Rata if >100%)"

    LEVERAGE_SCALAR_FULL = 2.5
    LEVERAGE_SCALAR_DEGRADED = 1.5
    MAX_WEIGHT_PER_ASSET = 0.70
    OVERALL_MULTIPLIER = 3.0

    def allocate(self, target_exposures: dict, market_data: dict) -> dict:
        active = {}
        for k, v in target_exposures.items():
            if v is not None and not (isinstance(v, float) and math.isnan(v)):
                vol = market_data.get(k, {}).get('vol_60d')
                if vol is not None and vol > 1e-10:
                    active[k] = {'exposure': v, 'vol': vol}

        if not active:
            return {k: 0.0 for k in target_exposures}

        bullish_count = sum(1 for info in active.values() if info['exposure'] > 0)
        leverage_scalar = self.LEVERAGE_SCALAR_DEGRADED if bullish_count <= 2 else self.LEVERAGE_SCALAR_FULL

        inv_vol = {asset: 1.0 / info['vol'] for asset, info in active.items()}
        total_inv_vol = sum(inv_vol.values())
        risk_weight = {asset: iv / total_inv_vol for asset, iv in inv_vol.items()}

        desired = {}
        for asset, info in active.items():
            desired[asset] = risk_weight[asset] * info['exposure'] * leverage_scalar

        for asset in desired:
            if desired[asset] > self.MAX_WEIGHT_PER_ASSET:
                desired[asset] = self.MAX_WEIGHT_PER_ASSET

        for asset in desired:
            desired[asset] = desired[asset] * self.OVERALL_MULTIPLIER

        total_desired = sum(desired.values())
        if total_desired > 1.0:
            scale = 1.0 / total_desired
            desired = {k: v * scale for k, v in desired.items()}

        result = {}
        for asset in target_exposures:
            result[asset] = desired.get(asset, 0.0)

        return result
