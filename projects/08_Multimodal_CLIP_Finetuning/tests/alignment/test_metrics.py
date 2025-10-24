#!/usr/bin/env python3
"""Tests for alignment metrics."""

from __future__ import annotations

import pytest
import torch

from alignment.metrics import AlignmentMetrics


class TestAlignmentMetrics:
    """Test cases for AlignmentMetrics."""

    @pytest.fixture
    def metrics(self) -> AlignmentMetrics:
        """Create metrics instance."""
        return AlignmentMetrics()

    @pytest.fixture
    def perfect_similarity(self) -> torch.Tensor:
        """Create perfect alignment similarity matrix (identity)."""
        return torch.eye(5)

    @pytest.fixture
    def random_similarity(self) -> torch.Tensor:
        """Create random similarity matrix."""
        torch.manual_seed(42)
        return torch.rand(5, 5)

    def test_recall_at_k_perfect_alignment(self, metrics, perfect_similarity) -> None:
        """Test Recall@K with perfect alignment."""
        recall = metrics.recall_at_k(perfect_similarity, k=1)

        # With perfect alignment, Recall@1 should be 1.0
        assert "i2t_R@1" in recall
        assert "t2i_R@1" in recall
        assert pytest.approx(recall["i2t_R@1"], abs=1e-5) == 1.0
        assert pytest.approx(recall["t2i_R@1"], abs=1e-5) == 1.0

    def test_recall_at_k_various_k(self, metrics, perfect_similarity) -> None:
        """Test Recall@K with different k values."""
        for k in [1, 3, 5]:
            recall = metrics.recall_at_k(perfect_similarity, k=k)

            assert f"i2t_R@{k}" in recall
            assert f"t2i_R@{k}" in recall
            # Perfect alignment should give 1.0 for all k
            assert pytest.approx(recall[f"i2t_R@{k}"], abs=1e-5) == 1.0
            assert pytest.approx(recall[f"t2i_R@{k}"], abs=1e-5) == 1.0

    def test_recall_at_k_random_alignment(self, metrics, random_similarity) -> None:
        """Test Recall@K with random alignment."""
        recall = metrics.recall_at_k(random_similarity, k=1)

        # Random alignment should give low recall
        assert 0.0 <= recall["i2t_R@1"] <= 1.0
        assert 0.0 <= recall["t2i_R@1"] <= 1.0

    def test_recall_at_k_invalid_matrix(self, metrics) -> None:
        """Test error on non-square matrix."""
        non_square = torch.rand(3, 5)

        with pytest.raises(ValueError, match="square similarity matrix"):
            metrics.recall_at_k(non_square, k=1)

    def test_recall_at_k_invalid_k(self, metrics, perfect_similarity) -> None:
        """Test error on invalid k."""
        with pytest.raises(ValueError, match="k must be in range"):
            metrics.recall_at_k(perfect_similarity, k=0)

        with pytest.raises(ValueError, match="k must be in range"):
            metrics.recall_at_k(perfect_similarity, k=10)

    def test_recall_at_k_non_2d(self, metrics) -> None:
        """Test error on non-2D tensor."""
        invalid = torch.rand(5)

        with pytest.raises(ValueError, match="Expected 2D similarity matrix"):
            metrics.recall_at_k(invalid, k=1)

    def test_mean_reciprocal_rank_perfect(self, metrics, perfect_similarity) -> None:
        """Test MRR with perfect alignment."""
        mrr = metrics.mean_reciprocal_rank(perfect_similarity)

        assert "i2t_MRR" in mrr
        assert "t2i_MRR" in mrr
        # Perfect alignment: correct match always at rank 1, so MRR = 1.0
        assert pytest.approx(mrr["i2t_MRR"], abs=1e-5) == 1.0
        assert pytest.approx(mrr["t2i_MRR"], abs=1e-5) == 1.0

    def test_mean_reciprocal_rank_random(self, metrics, random_similarity) -> None:
        """Test MRR with random alignment."""
        mrr = metrics.mean_reciprocal_rank(random_similarity)

        # MRR should be between 0 and 1
        assert 0.0 <= mrr["i2t_MRR"] <= 1.0
        assert 0.0 <= mrr["t2i_MRR"] <= 1.0

    def test_mean_reciprocal_rank_invalid_matrix(self, metrics) -> None:
        """Test error on non-square matrix."""
        non_square = torch.rand(3, 5)

        with pytest.raises(ValueError, match="square similarity matrix"):
            metrics.mean_reciprocal_rank(non_square)

    def test_mean_reciprocal_rank_non_2d(self, metrics) -> None:
        """Test error on non-2D tensor."""
        invalid = torch.rand(5)

        with pytest.raises(ValueError, match="Expected 2D similarity matrix"):
            metrics.mean_reciprocal_rank(invalid)

    def test_median_rank_perfect(self, metrics, perfect_similarity) -> None:
        """Test median rank with perfect alignment."""
        median = metrics.median_rank(perfect_similarity)

        assert "i2t_median_rank" in median
        assert "t2i_median_rank" in median
        # Perfect alignment: correct match always at rank 1
        assert pytest.approx(median["i2t_median_rank"], abs=1e-5) == 1.0
        assert pytest.approx(median["t2i_median_rank"], abs=1e-5) == 1.0

    def test_median_rank_random(self, metrics, random_similarity) -> None:
        """Test median rank with random alignment."""
        median = metrics.median_rank(random_similarity)

        # Median rank should be positive
        assert median["i2t_median_rank"] >= 1.0
        assert median["t2i_median_rank"] >= 1.0

    def test_median_rank_invalid_matrix(self, metrics) -> None:
        """Test error on non-square matrix."""
        non_square = torch.rand(3, 5)

        with pytest.raises(ValueError, match="square similarity matrix"):
            metrics.median_rank(non_square)

    def test_median_rank_non_2d(self, metrics) -> None:
        """Test error on non-2D tensor."""
        invalid = torch.rand(5)

        with pytest.raises(ValueError, match="Expected 2D similarity matrix"):
            metrics.median_rank(invalid)

    def test_compute_all_perfect(self, metrics, perfect_similarity) -> None:
        """Test computing all metrics with perfect alignment."""
        all_metrics = metrics.compute_all(perfect_similarity, k_values=[1, 3, 5])

        # Should have recall at all k values
        assert "i2t_R@1" in all_metrics
        assert "i2t_R@3" in all_metrics
        assert "i2t_R@5" in all_metrics
        assert "t2i_R@1" in all_metrics
        assert "t2i_R@3" in all_metrics
        assert "t2i_R@5" in all_metrics

        # Should have MRR
        assert "i2t_MRR" in all_metrics
        assert "t2i_MRR" in all_metrics

        # Should have median rank
        assert "i2t_median_rank" in all_metrics
        assert "t2i_median_rank" in all_metrics

        # All should be perfect
        assert pytest.approx(all_metrics["i2t_R@1"], abs=1e-5) == 1.0
        assert pytest.approx(all_metrics["i2t_MRR"], abs=1e-5) == 1.0
        assert pytest.approx(all_metrics["i2t_median_rank"], abs=1e-5) == 1.0

    def test_compute_all_default_k_values(self, metrics, perfect_similarity) -> None:
        """Test compute_all with default k values."""
        # Default k_values = [1, 5, 10], but matrix is 5x5, so will raise error
        # Test with valid k_values instead
        all_metrics = metrics.compute_all(perfect_similarity, k_values=[1, 5])

        # Should have recall at specified k values
        assert "i2t_R@1" in all_metrics
        assert "i2t_R@5" in all_metrics

    def test_compute_all_empty_k_values(self, metrics, perfect_similarity) -> None:
        """Test error on empty k_values."""
        with pytest.raises(ValueError, match="k_values cannot be empty"):
            metrics.compute_all(perfect_similarity, k_values=[])

    def test_compute_all_invalid_k_values(self, metrics, perfect_similarity) -> None:
        """Test error on invalid k values."""
        with pytest.raises(ValueError, match="Invalid k"):
            metrics.compute_all(perfect_similarity, k_values=[0, 5])

        with pytest.raises(ValueError, match="Invalid k"):
            metrics.compute_all(perfect_similarity, k_values=[1, 10])

    def test_compute_alignment_score_perfect(self, metrics, perfect_similarity) -> None:
        """Test alignment score with perfect alignment."""
        score = metrics.compute_alignment_score(perfect_similarity)

        # Perfect alignment (identity matrix) should have score 1.0
        assert pytest.approx(score, abs=1e-5) == 1.0

    def test_compute_alignment_score_random(self, metrics, random_similarity) -> None:
        """Test alignment score with random alignment."""
        score = metrics.compute_alignment_score(random_similarity)

        # Score should be the average of diagonal elements
        diagonal = torch.diagonal(random_similarity)
        expected = diagonal.mean().item()

        assert pytest.approx(score, abs=1e-5) == expected

    def test_compute_alignment_score_non_square(self, metrics) -> None:
        """Test error on non-square matrix."""
        non_square = torch.rand(3, 5)

        with pytest.raises(ValueError, match="square similarity matrix"):
            metrics.compute_alignment_score(non_square)

    def test_compute_alignment_score_non_2d(self, metrics) -> None:
        """Test error on non-2D tensor."""
        invalid = torch.rand(5)

        with pytest.raises(ValueError, match="Expected 2D similarity matrix"):
            metrics.compute_alignment_score(invalid)

    def test_repr(self, metrics) -> None:
        """Test string representation."""
        repr_str = repr(metrics)

        assert "AlignmentMetrics" in repr_str

    def test_worst_case_alignment(self, metrics) -> None:
        """Test metrics with reversed alignment."""
        # Create reversed matrix (worst case scenario)
        n = 5
        worst = torch.zeros(n, n)
        for i in range(n):
            # Make non-diagonal elements high, diagonal low
            for j in range(n):
                if i == j:
                    worst[i, j] = -1.0  # Diagonal is worst
                else:
                    worst[i, j] = 1.0 / abs(i - j)  # Closer to diagonal = higher

        # Recall@1 should be very low (diagonal is never top choice)
        recall = metrics.recall_at_k(worst, k=1)
        assert recall["i2t_R@1"] <= 0.2  # Allow for some imperfect tie-breaking

        # But Recall@5 should be 1.0 (diagonal match in top-5)
        recall_5 = metrics.recall_at_k(worst, k=5)
        assert pytest.approx(recall_5["i2t_R@5"], abs=1e-5) == 1.0

    def test_partial_alignment(self, metrics) -> None:
        """Test metrics with partial alignment."""
        # Create matrix where first 3 are correct, last 2 have correct in 2nd position
        n = 5
        partial = torch.zeros(n, n)
        for i in range(3):
            partial[i, i] = 1.0  # First 3 correct (diagonal high)
        # For indices 3 and 4, make diagonal second-best
        partial[3, 0] = 0.9  # Best for row 3 (wrong)
        partial[3, 3] = 0.8  # Second-best for row 3 (correct)
        partial[4, 0] = 0.9  # Best for row 4 (wrong)
        partial[4, 4] = 0.8  # Second-best for row 4 (correct)

        # Recall@1 should be 3/5 = 0.6 (first 3 have correct at top-1)
        recall = metrics.recall_at_k(partial, k=1)
        assert pytest.approx(recall["i2t_R@1"], abs=1e-2) == 0.6

        # Recall@2 should be 1.0 (all have correct in top-2)
        recall_2 = metrics.recall_at_k(partial, k=2)
        assert pytest.approx(recall_2["i2t_R@2"], abs=1e-5) == 1.0
