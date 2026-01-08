"""NoOpKVCacheManager for non-attention models.

Provides a minimal implementation of the KVCacheManager interface for models
that don't use KV cache (e.g., audio tokenizers, diffusion decoders).

This allows vLLM's scheduler to work with non-attention models without
requiring HybridKVCacheCoordinator which asserts on empty attention groups.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from vllm.logger import init_logger
from vllm.v1.core.kv_cache_manager import KVCacheBlocks

if TYPE_CHECKING:
    from vllm.v1.request import Request

logger = init_logger(__name__)


class NoOpKVCacheBlocks(KVCacheBlocks):
    """No-op KV cache blocks for non-attention models."""
    
    def __init__(self):
        # Empty block_ids list for each KV cache group (we have none)
        self._block_ids: list[list[int]] = [[]]
    
    def get_block_ids(self) -> list[list[int]]:
        return self._block_ids


class NoOpKVCacheManager:
    """No-op KV cache manager for non-attention models.
    
    Used by OmniGenerationScheduler when kv_cache_groups is empty.
    Provides the same interface as KVCacheManager but does nothing.
    """
    
    def __init__(self, log_stats: bool = False):
        self.log_stats = log_stats
        self._no_op_blocks = NoOpKVCacheBlocks()
        logger.info("Initialized NoOpKVCacheManager for non-attention model")
    
    def allocate_slots(
        self,
        request: Request,
        num_new_tokens: int,
        num_lookahead_tokens: int = 0,
    ) -> KVCacheBlocks | None:
        """Return no-op blocks - no actual allocation needed."""
        return self._no_op_blocks
    
    def free(self, request: Request) -> None:
        """No-op - nothing to free."""
        pass
    
    def get_num_common_prefix_blocks(self, request_id: str) -> list[int]:
        """Return empty prefix blocks list."""
        return [0]
    
    def take_events(self) -> list | None:
        """Return no events."""
        return None
    
    def reset_prefix_cache(self) -> bool:
        """No-op reset."""
        return True
    
    def get_prefix_cache_stats(self):
        """Return empty stats."""
        return None
    
    def get_num_free_blocks(self) -> int:
        """Return large number to indicate no memory pressure."""
        return 1000000
    
    def get_num_total_blocks(self) -> int:
        """Return same as free blocks."""
        return 1000000
    
    def get_num_cached_blocks(self) -> int:
        """No cached blocks."""
        return 0
