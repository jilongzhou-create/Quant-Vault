import math
from portfolio_allocators.base_allocator import BaseAllocator
from portfolio_allocators import register_allocator


@register_allocator
class V2DynamicScalarWeight(BaseAllocator):
    name = "V2DynamicScalarWeight"
    description = "Dynamic base quota * exposure * LEVERAGE_SCALAR(2.5) + Pro-Rata cap, no margin"

    LEVERAGE_SCALAR = 2.5

    def allocate(self, target_exposures: dict, market_data: dict) -> dict:
        active = {
            k: v for k, v in target_exposures.items()
            if v is not None and not (isinstance(v, float) and math.isnan(v))
        }

        if not active:
            return {k: 0.0 for k in target_exposures}

        n_active = len(active)
        base_quota = 1.0 / n_active

        desired = {}
        for asset, exposure in active.items():
            desired[asset] = base_quota * exposure * self.LEVERAGE_SCALAR

        total_desired = sum(desired.values())

        if total_desired > 1.0:
            scale = 1.0 / total_desired
            desired = {k: v * scale for k, v in desired.items()}

        result = {}
        for asset in target_exposures:
            result[asset] = desired.get(asset, 0.0)

        return result
