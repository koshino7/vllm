# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
TD-Pipe: Temporally-Disaggregated Pipeline Parallelism Scheduler (Simplified)

A simplified implementation for vLLM V1 that focuses on:
1. Phase separation (prefill vs decode) based on KV cache usage
2. Minimizing phase switching overhead
3. Filling KV cache as much as possible before switching

Note: This simplified version does NOT use profiler CSV data. Instead, it uses
KV cache usage as the primary metric for scheduling decisions.
"""

import math
import time
from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from vllm import envs
from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.v1.core.sched.scheduler import Scheduler
    from vllm.v1.request import Request

logger = init_logger(__name__)


@dataclass
class TDPipeConfig:
    """Simplified configuration for TD-Pipe scheduler."""
    # KV cache threshold to switch from prefill to decode (0.0-1.0)
    # When KV cache usage exceeds this, switch to decode phase
    prefill_to_decode_threshold: float = 0.85
    
    # KV cache threshold to switch back from decode to prefill (0.0-1.0)
    # When KV cache usage drops below this and there are waiting requests,
    # switch back to prefill phase
    decode_to_prefill_threshold: float = 0.50
    
    # Minimum number of decode steps before considering switching back to prefill
    min_decode_steps: int = 1
    
    # Maximum number of consecutive prefill batches (safety limit)
    max_consecutive_prefills: int = 100


class TDPipeSchedulerMixin:
    """
    Simplified TD-Pipe scheduler mixin for vLLM V1.
    
    Key design decisions:
    1. NO chunked prefill - TD-Pipe works best with complete prefills
    2. Phase separation - either in PREFILL mode or DECODE mode, not mixed
    3. KV cache driven - use KV cache usage to determine phase transitions
    
    State Machine:
    - PREFILL_PHASE: Execute new requests (waiting queue) until KV cache is full
                     or no more waiting requests, then switch to DECODE
    - DECODE_PHASE: Execute running requests until some complete and KV cache
                    drops below threshold, then switch back to PREFILL
    """
    
    def __init__(self, scheduler: "Scheduler"):
        self.scheduler = scheduler
        self.config = TDPipeConfig()
        
        # TD-Pipe state
        self.td_pipe_enabled = envs.VLLM_USE_TD_PIPE
        self.prefill_mode = True  # Start in prefill mode
        self.decode_steps = 0
        self.prefill_batches = 0
        
        # Track if we're in the middle of a phase transition
        self.transitioning = False
        
        logger.info(
            f"TD-Pipe scheduler initialized (enabled={self.td_pipe_enabled}, "
            f"prefill_threshold={self.config.prefill_to_decode_threshold}, "
            f"decode_threshold={self.config.decode_to_prefill_threshold})"
        )
    
    def _get_kv_cache_usage(self) -> float:
        """Get current KV cache usage ratio (0.0-1.0)."""
        # Use the kv_cache_manager's usage property if available
        if hasattr(self.scheduler.kv_cache_manager, 'usage'):
            return self.scheduler.kv_cache_manager.usage
        
        # Fallback: calculate from num_gpu_blocks
        total_blocks = self.scheduler.cache_config.num_gpu_blocks
        if total_blocks == 0:
            return 0.0
        
        # Estimate usage from running requests
        # This is a rough estimate; actual implementation may vary
        return 0.0  # Will be overridden by actual logic below
    
    def _should_stay_in_prefill(self) -> bool:
        """Determine if we should stay in prefill mode."""
        scheduler = self.scheduler
        
        # If no waiting requests, switch to decode
        if len(scheduler.waiting) == 0:
            logger.debug("TD-Pipe: No waiting requests, switching to decode")
            return False
        
        # Check KV cache usage
        kv_usage = self._get_kv_cache_usage()
        if kv_usage >= self.config.prefill_to_decode_threshold:
            logger.debug(f"TD-Pipe: KV cache full ({kv_usage:.2f}), switching to decode")
            return False
        
        # Safety limit on consecutive prefill batches
        if self.prefill_batches >= self.config.max_consecutive_prefills:
            logger.debug("TD-Pipe: Max prefill batches reached, switching to decode")
            return False
        
        return True
    
    def _should_switch_to_prefill(self) -> bool:
        """Determine if we should switch from decode back to prefill."""
        scheduler = self.scheduler
        
        # Only switch if we have waiting requests
        if len(scheduler.waiting) == 0:
            return False
        
        # Minimum decode steps before switching (avoid thrashing)
        if self.decode_steps < self.config.min_decode_steps:
            return False
        
        # Check KV cache usage
        kv_usage = self._get_kv_cache_usage()
        if kv_usage < self.config.decode_to_prefill_threshold:
            logger.debug(
                f"TD-Pipe: KV cache freed ({kv_usage:.2f}), "
                f"switching back to prefill"
            )
            return True
        
        return False
    
    def update_mode(self) -> bool:
        """
        Update scheduling mode (prefill vs decode).
        
        Returns True if in prefill mode, False if in decode mode.
        """
        if not self.td_pipe_enabled:
            return False  # Let default scheduler handle everything
        
        if self.prefill_mode:
            # Check if we should switch to decode
            if not self._should_stay_in_prefill():
                self.prefill_mode = False
                self.decode_steps = 0
                self.prefill_batches = 0
                logger.info("TD-Pipe: Switching PREFILL -> DECODE")
        else:
            # Check if we should switch back to prefill
            if self._should_switch_to_prefill():
                self.prefill_mode = True
                self.prefill_batches = 0
                logger.info("TD-Pipe: Switching DECODE -> PREFILL")
        
        return self.prefill_mode
    
    def on_prefill_batch_scheduled(self, num_requests: int):
        """Called when a batch of prefill requests is scheduled."""
        if self.td_pipe_enabled:
            self.prefill_batches += 1
    
    def on_decode_step_completed(self):
        """Called when a decode step is completed."""
        if self.td_pipe_enabled:
            self.decode_steps += 1
    
    def get_current_mode(self) -> str:
        """Get current mode as string for logging."""
        if not self.td_pipe_enabled:
            return "DISABLED"
        return "PREFILL" if self.prefill_mode else "DECODE"


# Helper function to check if a request is in prefill phase
def is_prefill_request(request: "Request") -> bool:
    """Check if a request is still in prefill phase."""
    # A request is in prefill phase if it hasn't computed all prompt tokens yet
    return request.num_computed_tokens < request.num_prompt_tokens