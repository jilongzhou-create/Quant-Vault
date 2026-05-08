import math
from portfolio_allocators.base_allocator import BaseAllocator
from portfolio_allocators import register_allocator


@register_allocator
class V1DynamicEqualWeight(BaseAllocator):
    name = "V1DynamicEqualWeight"
    description = "Surviving assets dynamic equal weight + LEVERAGE_SCALAR amplification + 1.0 exposure hard cap"

    def __init__(self, leverage_scalar: float = 2.5):
        self.leverage_scalar = leverage_scalar

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
            desired[asset] = base_quota * exposure * self.leverage_scalar

        total_desired = sum(desired.values())

        if total_desired > 1.0:
            scale = 1.0 / total_desired
            desired = {k: v * scale for k, v in desired.items()}

        result = {}
        for asset in target_exposures:
            if asset in desired:
                result[asset] = min(desired[asset], 1.0)
            else:
                result[asset] = 0.0

        return result
