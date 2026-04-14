"""
DPO preference pair construction and export utilities.

Constructs (chosen, rejected) preference pairs from multi-persona signals
ranked by computed rewards for DPO fine-tuning.
"""

from dataclasses import dataclass

from loguru import logger

from swarm.training_capture import TrainingExample
from training.reward_engine import ComputedReward
from verifier.outcome import VerifiedOutcome


@dataclass(slots=True, frozen=True)
class PreferencePair:
    """
    DPO preference pair with shared prompt and ranked responses.

    Both chosen and rejected signals are generated from the same prompt/context,
    ensuring valid preference learning signal.
    """

    # Shared input (identical for chosen and rejected)
    prompt: str  # Task prompt (market analysis question)
    context_id: str  # Groups examples from same market context

    # Chosen response (higher reward)
    chosen_reasoning: str
    chosen_direction: str
    chosen_confidence: float
    chosen_reward: float
    chosen_example_id: str
    chosen_persona: str

    # Rejected response (lower reward)
    rejected_reasoning: str
    rejected_direction: str
    rejected_confidence: float
    rejected_reward: float
    rejected_example_id: str
    rejected_persona: str

    # Metadata
    reward_delta: float  # chosen_reward - rejected_reward
    symbol: str
    timestamp_ms: int
    market_regime: str


def compute_reward_delta(chosen_reward: float, rejected_reward: float) -> float:
    """
    Compute reward delta between chosen and rejected signals.

    Args:
        chosen_reward: Reward for chosen signal
        rejected_reward: Reward for rejected signal

    Returns:
        Delta (chosen - rejected), always positive
    """
    return chosen_reward - rejected_reward


def validate_preference_pair(
    chosen_example: TrainingExample,
    rejected_example: TrainingExample,
    chosen_reward: ComputedReward,
    rejected_reward: ComputedReward,
    min_delta: float = 0.2,
) -> tuple[bool, str]:
    """
    Validate preference pair meets quality criteria.

    Requirements:
    1. Same context_id (same prompt/market context)
    2. Sufficient reward delta (min_delta threshold)
    3. Chosen reward > rejected reward
    4. Both examples have valid signals

    Args:
        chosen_example: Example with higher reward
        rejected_example: Example with lower reward
        chosen_reward: Computed reward for chosen
        rejected_reward: Computed reward for rejected
        min_delta: Minimum reward delta threshold

    Returns:
        Tuple of (is_valid, reason)
    """
    # Same context check
    if chosen_example.context_id != rejected_example.context_id:
        return False, f"Context ID mismatch: {chosen_example.context_id} != {rejected_example.context_id}"

    if not chosen_example.context_id:
        return False, "Empty context_id (examples from single-persona workflow)"

    # Reward ordering check
    delta = compute_reward_delta(chosen_reward.final_reward, rejected_reward.final_reward)
    if delta <= 0:
        return False, f"Invalid reward ordering: chosen={chosen_reward.final_reward:.3f}, rejected={rejected_reward.final_reward:.3f}"

    # Minimum delta check
    if delta < min_delta:
        return False, f"Reward delta {delta:.3f} below minimum {min_delta}"

    # Valid signals check
    if not chosen_example.generator_signal or not rejected_example.generator_signal:
        return False, "Missing generator signal"

    return True, f"Valid pair with delta={delta:.3f}"


def construct_preference_pairs(
    examples_with_rewards: list[tuple[TrainingExample, VerifiedOutcome, ComputedReward]],
    min_delta: float = 0.2,
    min_personas_per_context: int = 3,
) -> list[PreferencePair]:
    """
    Construct DPO preference pairs from multi-persona signals.

    Workflow:
    1. Group examples by context_id
    2. Filter contexts with insufficient persona diversity
    3. Rank signals within each context by reward
    4. Create (chosen, rejected) pairs with sufficient delta
    5. Validate each pair

    Args:
        examples_with_rewards: List of (example, outcome, reward) tuples
        min_delta: Minimum reward delta for valid pairs
        min_personas_per_context: Minimum personas required per context

    Returns:
        List of validated preference pairs

    Example:
        >>> pairs = construct_preference_pairs(
        ...     examples_with_rewards,
        ...     min_delta=0.2,
        ...     min_personas_per_context=3,
        ... )
        >>> print(f"Constructed {len(pairs)} preference pairs")
    """
    from collections import defaultdict

    # Group by context_id
    context_groups: dict[str, list[tuple[TrainingExample, VerifiedOutcome, ComputedReward]]] = defaultdict(list)

    for example, outcome, reward in examples_with_rewards:
        if example.context_id:  # Skip single-persona examples
            context_groups[example.context_id].append((example, outcome, reward))

    preference_pairs: list[PreferencePair] = []

    # Process each context
    for context_id, group in context_groups.items():
        # Check persona diversity
        personas = {ex.persona for ex, _, _ in group}
        if len(personas) < min_personas_per_context:
            logger.warning(
                "Insufficient persona diversity",
                context_id=context_id,
                personas=len(personas),
                required=min_personas_per_context,
            )
            continue

        # Rank by reward (descending)
        sorted_group = sorted(
            group,
            key=lambda x: x[2].final_reward,  # ComputedReward.final_reward
            reverse=True,
        )

        # Create pairs: best vs worst, second-best vs second-worst, etc.
        # This maximizes reward delta while ensuring diversity
        num_pairs = len(sorted_group) // 2

        for i in range(num_pairs):
            chosen_example, chosen_outcome, chosen_reward = sorted_group[i]
            rejected_example, rejected_outcome, rejected_reward = sorted_group[-(i+1)]

            # Validate pair
            is_valid, reason = validate_preference_pair(
                chosen_example,
                rejected_example,
                chosen_reward,
                rejected_reward,
                min_delta=min_delta,
            )

            if not is_valid:
                logger.debug(
                    "Skipping invalid pair",
                    context_id=context_id,
                    reason=reason,
                )
                continue

            # Extract reasoning from signals
            chosen_reasoning = chosen_example.generator_signal.get("reasoning", "")
            rejected_reasoning = rejected_example.generator_signal.get("reasoning", "")

            if not chosen_reasoning or not rejected_reasoning:
                logger.warning(
                    "Missing reasoning field",
                    context_id=context_id,
                    chosen_has_reasoning=bool(chosen_reasoning),
                    rejected_has_reasoning=bool(rejected_reasoning),
                )
                continue

            # Create preference pair
            pair = PreferencePair(
                prompt=chosen_example.task_prompt,
                context_id=context_id,
                chosen_reasoning=chosen_reasoning,
                chosen_direction=chosen_example.generator_signal.get("direction", ""),
                chosen_confidence=chosen_example.generator_signal.get("confidence", 0.0),
                chosen_reward=chosen_reward.final_reward,
                chosen_example_id=chosen_example.example_id,
                chosen_persona=chosen_example.persona,
                rejected_reasoning=rejected_reasoning,
                rejected_direction=rejected_example.generator_signal.get("direction", ""),
                rejected_confidence=rejected_example.generator_signal.get("confidence", 0.0),
                rejected_reward=rejected_reward.final_reward,
                rejected_example_id=rejected_example.example_id,
                rejected_persona=rejected_example.persona,
                reward_delta=compute_reward_delta(
                    chosen_reward.final_reward,
                    rejected_reward.final_reward,
                ),
                symbol=chosen_example.symbol,
                timestamp_ms=chosen_example.timestamp_ms,
                market_regime=chosen_example.market_regime,
            )

            preference_pairs.append(pair)

    logger.info(
        "Preference pairs constructed",
        total_examples=len(examples_with_rewards),
        contexts=len(context_groups),
        pairs=len(preference_pairs),
    )

    return preference_pairs


def export_to_huggingface_format(pairs: list[PreferencePair]) -> list[dict]:
    """
    Export preference pairs to HuggingFace DPO dataset format.

    Format:
    {
        "prompt": "<market analysis task>",
        "chosen": "<reasoning for better signal>",
        "rejected": "<reasoning for worse signal>",
    }

    Args:
        pairs: List of validated preference pairs

    Returns:
        List of dictionaries in HuggingFace format

    Example:
        >>> hf_data = export_to_huggingface_format(pairs)
        >>> print(hf_data[0])
        {
            "prompt": "Analyze BTC/USDT 1h chart...",
            "chosen": "Strong momentum with RSI 65...",
            "rejected": "Expecting mean reversion but...",
        }
    """
    hf_dataset = []

    for pair in pairs:
        hf_dataset.append({
            "prompt": pair.prompt,
            "chosen": pair.chosen_reasoning,
            "rejected": pair.rejected_reasoning,
        })

    logger.info(
        "Exported to HuggingFace format",
        pairs=len(pairs),
    )

    return hf_dataset


def export_to_jsonl(
    pairs: list[PreferencePair],
    output_path: str,
    include_metadata: bool = True,
) -> int:
    """
    Export preference pairs to JSONL file.

    Args:
        pairs: List of preference pairs
        output_path: Path to output JSONL file
        include_metadata: If True, includes full metadata. If False, only HF format.

    Returns:
        Number of pairs exported

    Example:
        >>> export_to_jsonl(pairs, "outputs/dpo_pairs.jsonl")
        Exported 142 preference pairs to outputs/dpo_pairs.jsonl
        142
    """
    import json
    from pathlib import Path

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w") as f:
        for pair in pairs:
            if include_metadata:
                # Full metadata for analysis
                from dataclasses import asdict
                data = asdict(pair)
            else:
                # HuggingFace format only
                data = {
                    "prompt": pair.prompt,
                    "chosen": pair.chosen_reasoning,
                    "rejected": pair.rejected_reasoning,
                }

            f.write(json.dumps(data) + "\n")

    logger.info(
        "Exported to JSONL",
        path=str(output_file),
        pairs=len(pairs),
        include_metadata=include_metadata,
    )

    return len(pairs)
