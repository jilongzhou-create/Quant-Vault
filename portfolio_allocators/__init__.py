from portfolio_allocators.base_allocator import BaseAllocator

ALLOCATOR_REGISTRY = {}


def register_allocator(cls):
    ALLOCATOR_REGISTRY[cls.name] = cls
    return cls


def get_allocator(name: str) -> type:
    if name not in ALLOCATOR_REGISTRY:
        available = ', '.join(ALLOCATOR_REGISTRY.keys()) or 'none'
        raise ValueError(f"Unknown allocator '{name}'. Available: {available}")
    return ALLOCATOR_REGISTRY[name]


def list_allocators() -> list:
    return list(ALLOCATOR_REGISTRY.keys())
