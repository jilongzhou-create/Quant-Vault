import math
from portfolio_allocators import register_allocator
from portfolio_allocators.base_allocator import BaseAllocator


@register_allocator
class V5CrossAssetMomentumAllocator(BaseAllocator):
    name = "V5CrossAssetMomentum"
    description = "Cross-Asset 12M-1M Momentum Selection + AI Exposure Control (Top-1 monthly lock)"

    def __init__(self):
        self.selected_asset = None

    def allocate(self, target_exposures: dict, market_data: dict) -> dict:
        is_eom = market_data.get('is_eom', False)

        if is_eom or self.selected_asset is None:
            best_asset = None
            best_mom = -math.inf

            for asset in target_exposures:
                asset_md = market_data.get(asset, {})
                mom = asset_md.get('mom')
                if mom is None or (isinstance(mom, float) and math.isnan(mom)):
                    continue
                if mom > best_mom:
                    best_mom = mom
                    best_asset = asset

            if best_asset is not None and best_mom > 0:
                self.selected_asset = best_asset
            else:
                self.selected_asset = None

        weights = {asset: 0.0 for asset in target_exposures}

        if self.selected_asset is not None:
            ai_exposure = target_exposures.get(self.selected_asset, 0.0)
            if ai_exposure is None or (isinstance(ai_exposure, float) and math.isnan(ai_exposure)):
                ai_exposure = 0.0
            weights[self.selected_asset] = float(ai_exposure)

        return weights
