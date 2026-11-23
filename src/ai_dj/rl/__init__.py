"""Phase 4: Audio-grounded GRPO environment and Tinker RL trainer for creative DJ mixing."""

from ai_dj.rl.grpo_env import (
    CREATIVE_STYLES,
    GRPO_SYSTEM_PROMPT,
    AudioTransitionTask,
    CreativeMixPlan,
    GRPORewardBreakdown,
    build_audio_grpo_tasks,
    format_audio_grpo_prompt,
    parse_creative_plan,
    score_creative_plan,
)
from ai_dj.rl.tinker_grpo import GRPOConfig, plan_and_render_electronic_set, run_tinker_grpo

__all__ = [
    "CREATIVE_STYLES",
    "GRPO_SYSTEM_PROMPT",
    "AudioTransitionTask",
    "CreativeMixPlan",
    "GRPOConfig",
    "GRPORewardBreakdown",
    "build_audio_grpo_tasks",
    "format_audio_grpo_prompt",
    "parse_creative_plan",
    "plan_and_render_electronic_set",
    "run_tinker_grpo",
    "score_creative_plan",
]
