import math
from portfolio_allocators.base_allocator import BaseAllocator
from portfolio_allocators import register_allocator


@register_allocator
class V3RiskParityAllocator(BaseAllocator):
    name = "V3RiskParityAllocator"
    description = "Inverse-volatility risk parity: inv_vol weighting * exposure * LEVERAGE_SCALAR(2.5) + Pro-Rata cap"

    LEVERAGE_SCALAR = 2.5

    def allocate(self, target_exposures: dict, market_data: dict) -> dict:
        active = {}
        for k, v in target_exposures.items():
            if v is not None and not (isinstance(v, float) and math.isnan(v)):
                vol = market_data.get(k, {}).get('vol_60d')
                if vol is not None and vol > 1e-10:
                    active[k] = {'exposure': v, 'vol': vol}

        if not active:
            return {k: 0.0 for k in target_exposures}

        inv_vol = {asset: 1.0 / info['vol'] for asset, info in active.items()}
        total_inv_vol = sum(inv_vol.values())

        risk_weight = {asset: iv / total_inv_vol for asset, iv in inv_vol.items()}

        desired = {}
        for asset, info in active.items():
            desired[asset] = risk_weight[asset] * info['exposure'] * self.LEVERAGE_SCALAR

        total_desired = sum(desired.values())

        if total_desired > 1.0:
            scale = 1.0 / total_desired
            desired = {k: v * scale for k, v in desired.items()}

        result = {}
        for asset in target_exposures:
            result[asset] = desired.get(asset, 0.0)

        return result
