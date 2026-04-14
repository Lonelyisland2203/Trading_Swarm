"""Tests for inference queue with resume capability."""

import json
import pytest
import pandas as pd
from unittest.mock import AsyncMock

from data.inference_queue import (
    InferenceJob,
    InferenceQueue,
    ProcessingStats,
)
from data.prompt_builder import TaskType
from data.regime_filter import MarketRegime
from swarm.training_capture import TrainingExample


@pytest.fixture
def sample_job():
    """Create sample inference job."""
    df = pd.DataFrame({
        "timestamp": [1704067200000 + i * 3_600_000 for i in range(100)],
        "open": [50000.0] * 100,
        "high": [50100.0] * 100,
        "low": [49900.0] * 100,
        "close": [50000.0 + i * 10 for i in range(100)],
        "volume": [100.0] * 100,
    })

    return InferenceJob(
        job_id="job-001",
        context_id="BTC/USDT_1h_1704067200000_predict_direction",
        symbol="BTC/USDT",
        timeframe="1h",
        timestamp_ms=1704067200000,
        ohlcv_df=df,
        market_regime=MarketRegime.NEUTRAL,
        task_type=TaskType.PREDICT_DIRECTION,
        task_prompt="Test prompt",
    )


@pytest.fixture
def sample_examples():
    """Create sample training examples."""
    examples = []
    for i in range(5):
        ex = TrainingExample(
            symbol="BTC/USDT",
            timeframe="1h",
            timestamp_ms=1704067200000,
            market_regime="NEUTRAL",
            persona=f"persona_{i}",
            context_id="test_context_1",
        )
        examples.append(ex)
    return examples


class TestInferenceQueue:
    """Test inference queue basic functionality."""

    def test_init_no_resume(self, tmp_path):
        """Test initialization without resume."""
        output_file = tmp_path / "examples.jsonl"
        queue = InferenceQueue(output_file, resume=False)

        assert queue.output_file == output_file
        assert len(queue.completed_contexts) == 0
        assert queue.max_retries == 2

    def test_init_with_resume_empty_file(self, tmp_path):
        """Test initialization with resume but file doesn't exist."""
        output_file = tmp_path / "examples.jsonl"
        queue = InferenceQueue(output_file, resume=True)

        assert len(queue.completed_contexts) == 0

    def test_init_no_resume_truncates_existing_file(self, tmp_path):
        """Test that resume=False removes stale output from a prior run."""
        output_file = tmp_path / "examples.jsonl"

        # Simulate a prior run leaving data behind
        output_file.write_text('{"context_id": "stale_ctx", "symbol": "BTC"}\n')
        assert output_file.exists()

        queue = InferenceQueue(output_file, resume=False)

        # File should be deleted so appends start fresh
        assert not output_file.exists()
        assert len(queue.completed_contexts) == 0

    def test_init_resume_preserves_existing_file(self, tmp_path):
        """Test that resume=True does NOT truncate the existing output file."""
        output_file = tmp_path / "examples.jsonl"

        # Write data that should survive resume init
        output_file.write_text('{"context_id": "keep_me"}\n')

        queue = InferenceQueue(output_file, resume=True)

        # File must still exist with its content
        assert output_file.exists()
        assert output_file.read_text().strip() == '{"context_id": "keep_me"}'

    def test_init_creates_output_directory(self, tmp_path):
        """Test output directory is created."""
        output_file = tmp_path / "subdir" / "examples.jsonl"
        queue = InferenceQueue(output_file, resume=False)

        assert output_file.parent.exists()


class TestResumeState:
    """Test resume state loading."""

    def test_load_complete_contexts(self, tmp_path, sample_examples):
        """Test loading complete contexts (5 personas each)."""
        output_file = tmp_path / "examples.jsonl"

        # Write complete context (5 examples, same context_id)
        with open(output_file, "w") as f:
            for ex in sample_examples:
                f.write(json.dumps(ex.to_dict()) + "\n")

        queue = InferenceQueue(output_file, resume=True)

        assert len(queue.completed_contexts) == 1
        assert "test_context_1" in queue.completed_contexts

    def test_load_incomplete_contexts(self, tmp_path, sample_examples):
        """Test incomplete contexts are not marked complete."""
        output_file = tmp_path / "examples.jsonl"

        # Write only 3 examples (incomplete context)
        with open(output_file, "w") as f:
            for ex in sample_examples[:3]:
                f.write(json.dumps(ex.to_dict()) + "\n")

        queue = InferenceQueue(output_file, resume=True)

        # Should not be in completed (needs all 5 personas)
        assert len(queue.completed_contexts) == 0

    def test_load_multiple_contexts(self, tmp_path):
        """Test loading multiple complete contexts."""
        output_file = tmp_path / "examples.jsonl"

        # Create 3 complete contexts
        with open(output_file, "w") as f:
            for ctx_idx in range(3):
                for persona_idx in range(5):
                    ex = TrainingExample(
                        symbol="BTC/USDT",
                        timeframe="1h",
                        timestamp_ms=1704067200000,
                        market_regime="NEUTRAL",
                        persona=f"persona_{persona_idx}",
                        context_id=f"context_{ctx_idx}",
                    )
                    f.write(json.dumps(ex.to_dict()) + "\n")

        queue = InferenceQueue(output_file, resume=True)

        assert len(queue.completed_contexts) == 3
        assert "context_0" in queue.completed_contexts
        assert "context_1" in queue.completed_contexts
        assert "context_2" in queue.completed_contexts

    def test_load_malformed_jsonl(self, tmp_path):
        """Test handles malformed JSONL gracefully."""
        output_file = tmp_path / "examples.jsonl"

        with open(output_file, "w") as f:
            f.write('{"valid": "json"}\n')
            f.write('invalid json line\n')
            f.write('{"another": "valid"}\n')

        # Should not crash
        queue = InferenceQueue(output_file, resume=True)
        assert queue.completed_contexts is not None


@pytest.mark.asyncio
class TestProcessing:
    """Test job processing."""

    async def test_save_examples(self, tmp_path, sample_examples):
        """Test examples are saved to JSONL."""
        output_file = tmp_path / "examples.jsonl"
        queue = InferenceQueue(output_file, resume=False)

        queue._save_examples(sample_examples)

        # Verify file contents
        with open(output_file) as f:
            lines = f.readlines()

        assert len(lines) == 5
        for line in lines:
            data = json.loads(line)
            assert "symbol" in data
            assert "context_id" in data

    async def test_process_single_job_success(
        self, tmp_path, sample_job, sample_examples, monkeypatch
    ):
        """Test successful single job processing."""
        # Mock run_multi_persona_workflow
        async def mock_workflow(*args, **kwargs):
            summary = {
                "workflow_status": "success",
                "signals_generated": 5,
                "signals_accepted": 3,
            }
            return summary, sample_examples

        monkeypatch.setattr(
            "data.inference_queue.run_multi_persona_workflow",
            mock_workflow,
        )

        queue = InferenceQueue(tmp_path / "out.jsonl", resume=False)
        success, examples = await queue._process_single_job(sample_job)

        assert success is True
        assert len(examples) == 5

    async def test_process_single_job_failure(self, tmp_path, sample_job, monkeypatch):
        """Test failed job processing."""
        # Mock workflow that fails
        async def mock_workflow(*args, **kwargs):
            summary = {
                "workflow_status": "all_failed",
                "signals_generated": 0,
                "errors": ["Model unavailable"],
            }
            return summary, []

        monkeypatch.setattr(
            "data.inference_queue.run_multi_persona_workflow",
            mock_workflow,
        )

        queue = InferenceQueue(tmp_path / "out.jsonl", resume=False)
        success, examples = await queue._process_single_job(sample_job)

        assert success is False
        assert len(examples) == 0

    async def test_process_all_skip_completed(
        self, tmp_path, sample_job, sample_examples, monkeypatch
    ):
        """Test process_all skips completed contexts on resume."""
        output_file = tmp_path / "examples.jsonl"

        # Write completed context
        with open(output_file, "w") as f:
            for ex in sample_examples:
                ex.context_id = sample_job.context_id
                f.write(json.dumps(ex.to_dict()) + "\n")

        # Mock workflow (should not be called)
        mock_workflow = AsyncMock()
        monkeypatch.setattr(
            "data.inference_queue.run_multi_persona_workflow",
            mock_workflow,
        )

        queue = InferenceQueue(output_file, resume=True)
        stats = await queue.process_all([sample_job])

        # Job should be skipped
        assert stats.skipped_resume == 1
        assert stats.completed == 0
        mock_workflow.assert_not_called()

    async def test_process_all_with_progress_callback(
        self, tmp_path, sample_job, sample_examples, monkeypatch
    ):
        """Test process_all invokes progress callback."""
        # Mock successful workflow
        async def mock_workflow(*args, **kwargs):
            summary = {"workflow_status": "success", "signals_generated": 5, "signals_accepted": 3}
            return summary, sample_examples

        monkeypatch.setattr(
            "data.inference_queue.run_multi_persona_workflow",
            mock_workflow,
        )

        # Track callback invocations
        callback_calls = []

        def progress_callback(context_id, duration, success):
            callback_calls.append((context_id, duration, success))

        queue = InferenceQueue(tmp_path / "out.jsonl", resume=False)
        await queue.process_all([sample_job], progress_callback=progress_callback)

        assert len(callback_calls) == 1
        assert callback_calls[0][0] == sample_job.context_id
        assert callback_calls[0][1] > 0  # duration
        assert callback_calls[0][2] is True  # success

    async def test_process_all_retry_logic(
        self, tmp_path, sample_job, sample_examples, monkeypatch
    ):
        """Test retry logic on job failure."""
        # Mock workflow that fails twice then succeeds
        call_count = 0

        async def mock_workflow(*args, **kwargs):
            nonlocal call_count
            call_count += 1

            if call_count <= 2:
                summary = {"workflow_status": "all_failed"}
                return summary, []
            else:
                summary = {"workflow_status": "success", "signals_generated": 5, "signals_accepted": 3}
                return summary, sample_examples

        monkeypatch.setattr(
            "data.inference_queue.run_multi_persona_workflow",
            mock_workflow,
        )

        queue = InferenceQueue(
            tmp_path / "out.jsonl",
            resume=False,
            max_retries=2,
            retry_delay_seconds=0,  # No delay for test
        )

        stats = await queue.process_all([sample_job])

        # Should succeed after retries
        assert stats.completed == 1
        assert stats.failed == 0
        assert call_count == 3  # 1 initial + 2 retries


class TestProcessingStats:
    """Test processing statistics."""

    def test_stats_creation(self):
        """Test creating processing stats."""
        stats = ProcessingStats(
            total_jobs=100,
            completed=80,
            failed=10,
            skipped_resume=10,
            total_examples=400,
            elapsed_seconds=1000.0,
        )

        assert stats.total_jobs == 100
        assert stats.completed == 80
        assert stats.failed == 10
        assert stats.total_examples == 400
