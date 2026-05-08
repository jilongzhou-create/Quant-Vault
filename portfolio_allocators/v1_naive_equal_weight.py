import math
from portfolio_allocators.base_allocator import BaseAllocator
from portfolio_allocators import register_allocator


@register_allocator
class V1NaiveEqualWeight(BaseAllocator):
    name = "V1NaiveEqualWeight"
    description = "Naive equal weight: 1.0 total capital split equally among surviving assets, no leverage"

    def allocate(self, target_exposures: dict, market_data: dict) -> dict:
        active = {
            k: v for k, v in target_exposures.items()
            if v is not None and not (isinstance(v, float) and math.isnan(v))
        }

        if not active:
            return {k: 0.0 for k in target_exposures}

        n_active = len(active)
        weight = 1.0 / n_active

        result = {}
        for asset in target_exposures:
            if asset in active:
                result[asset] = weight
            else:
                result[asset] = 0.0

        return result
