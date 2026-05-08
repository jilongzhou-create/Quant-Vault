from abc import ABC, abstractmethod


class BaseAllocator(ABC):
    name: str = "BaseAllocator"
    description: str = ""

    @abstractmethod
    def allocate(self, target_exposures: dict, market_data: dict) -> dict:
        """
        Compute final portfolio weight allocation.

        Args:
            target_exposures: dict mapping asset name (e.g. 'GOLD', 'TLT', 'SPY', 'BTC')
                              to its target exposure value (float or NaN).
                              NaN means the asset has no data on this date (e.g. BTC before 2018).
            market_data: dict mapping asset name to a dict of auxiliary market data,
                         e.g. {'GOLD': {'price': 1800, 'return': 0.01, 'vol_20d': 0.12}, ...}

        Returns:
            dict mapping asset name to its final portfolio weight (float).
            All weights should be non-negative. The allocator decides whether
            the sum of weights can exceed 1.0 (leverage) or must be <= 1.0.
        """
        raise NotImplementedError
