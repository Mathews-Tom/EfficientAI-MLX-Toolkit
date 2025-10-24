#!/usr/bin/env python3
"""Tests for temperature scheduler."""

from __future__ import annotations

import pytest

from src.losses.temperature_scheduler import TemperatureScheduler


class TestTemperatureScheduler:
    """Tests for TemperatureScheduler."""

    def test_initialization_default(self):
        """Test scheduler initialization with defaults."""
        scheduler = TemperatureScheduler()

        assert scheduler.initial_temp == 0.07
        assert scheduler.min_temp == 0.01
        assert scheduler.max_temp == 0.1
        assert scheduler.warmup_steps == 1000
        assert scheduler.schedule_type == "warmup"
        assert scheduler.current_temp == 0.07
        assert scheduler.current_step == 0

    def test_initialization_custom(self):
        """Test initialization with custom parameters."""
        scheduler = TemperatureScheduler(
            initial_temp=0.05,
            min_temp=0.01,
            max_temp=0.2,
            warmup_steps=500,
            schedule_type="cosine",
            total_steps=10000,
        )

        assert scheduler.initial_temp == 0.05
        assert scheduler.min_temp == 0.01
        assert scheduler.max_temp == 0.2
        assert scheduler.warmup_steps == 500
        assert scheduler.schedule_type == "cosine"

    def test_initialization_invalid_params(self):
        """Test initialization with invalid parameters."""
        # Invalid initial temperature
        with pytest.raises(ValueError, match="Initial temperature must be positive"):
            TemperatureScheduler(initial_temp=-0.01)

        # Invalid min temperature
        with pytest.raises(ValueError, match="Minimum temperature must be positive"):
            TemperatureScheduler(min_temp=0.0)

        # Invalid max temperature
        with pytest.raises(ValueError, match="Maximum temperature must be positive"):
            TemperatureScheduler(max_temp=-0.1)

        # Min > Max
        with pytest.raises(ValueError, match="Minimum temperature.*cannot be greater"):
            TemperatureScheduler(min_temp=0.5, max_temp=0.1)

        # Initial not in range
        with pytest.raises(ValueError, match="Initial temperature.*must be between"):
            TemperatureScheduler(initial_temp=0.5, min_temp=0.01, max_temp=0.1)

        # Negative warmup steps
        with pytest.raises(ValueError, match="Warmup steps must be non-negative"):
            TemperatureScheduler(warmup_steps=-100)

        # Invalid schedule type
        with pytest.raises(ValueError, match="Schedule type must be one of"):
            TemperatureScheduler(schedule_type="invalid")

        # Missing total_steps for cosine
        with pytest.raises(ValueError, match="total_steps must be specified"):
            TemperatureScheduler(schedule_type="cosine")

        # Invalid decay rate
        with pytest.raises(ValueError, match="Decay rate must be in"):
            TemperatureScheduler(schedule_type="exponential", total_steps=1000, decay_rate=1.5)

    def test_constant_schedule(self):
        """Test constant temperature schedule."""
        scheduler = TemperatureScheduler(
            initial_temp=0.07,
            schedule_type="constant",
        )

        # Temperature should remain constant
        for step in range(100):
            temp = scheduler.step(step)
            assert temp == 0.07

    def test_warmup_schedule(self):
        """Test warmup temperature schedule."""
        scheduler = TemperatureScheduler(
            initial_temp=0.07,
            max_temp=0.1,
            warmup_steps=100,
            schedule_type="warmup",
        )

        # During warmup, temperature should decrease from max to initial
        temp_start = scheduler.step(0)
        assert temp_start == 0.1  # Should start at max_temp

        temp_mid = scheduler.step(50)
        assert 0.07 < temp_mid < 0.1  # Should be between initial and max

        temp_end = scheduler.step(100)
        assert abs(temp_end - 0.07) < 1e-6  # Should reach initial_temp

        # After warmup, should stay at initial_temp
        temp_after = scheduler.step(200)
        assert temp_after == 0.07

    def test_cosine_schedule(self):
        """Test cosine annealing schedule."""
        scheduler = TemperatureScheduler(
            initial_temp=0.07,
            min_temp=0.01,
            max_temp=0.1,
            warmup_steps=100,
            total_steps=1000,
            schedule_type="cosine",
        )

        # During warmup
        temp_warmup = scheduler.step(50)
        assert 0.07 < temp_warmup < 0.1

        # After warmup, should start cosine decay
        temp_mid = scheduler.step(550)
        assert 0.01 < temp_mid < 0.07

        # At end, should approach min_temp
        temp_end = scheduler.step(1000)
        assert temp_end <= 0.07
        assert temp_end >= 0.01

    def test_exponential_schedule(self):
        """Test exponential decay schedule."""
        scheduler = TemperatureScheduler(
            initial_temp=0.07,
            min_temp=0.01,
            warmup_steps=100,
            total_steps=1000,
            schedule_type="exponential",
            decay_rate=0.99,
        )

        # During warmup
        temp_warmup = scheduler.step(50)
        assert temp_warmup > 0.07

        # After warmup, should decay exponentially
        temp_mid = scheduler.step(550)
        assert temp_mid < 0.07
        assert temp_mid > 0.01

        # Should approach min_temp asymptotically
        temp_far = scheduler.step(5000)
        assert temp_far >= 0.01  # Clipped to min_temp

    def test_adaptive_schedule(self):
        """Test adaptive temperature schedule."""
        scheduler = TemperatureScheduler(
            initial_temp=0.07,
            min_temp=0.01,
            max_temp=0.1,
            warmup_steps=100,
            schedule_type="adaptive",
        )

        # Need to provide loss values for adaptive scheduling
        # Simulate increasing loss (should increase temperature)
        for step in range(100, 300):
            loss = 2.0 + step * 0.01  # Increasing loss
            temp = scheduler.step(step, loss=loss)

        # After loss increase, temperature should have increased
        assert scheduler.current_temp > 0.07

        # Simulate decreasing loss (should decrease temperature)
        for step in range(300, 500):
            loss = 3.0 - (step - 300) * 0.01  # Decreasing loss
            temp = scheduler.step(step, loss=loss)

    def test_adaptive_requires_loss(self):
        """Test that adaptive scheduling requires loss value."""
        scheduler = TemperatureScheduler(
            initial_temp=0.07,
            warmup_steps=100,
            schedule_type="adaptive",
        )

        # During warmup, loss not required
        scheduler.step(50)

        # After warmup, loss required
        with pytest.raises(ValueError, match="Loss must be provided"):
            scheduler.step(150)  # No loss provided

    def test_temperature_bounds(self):
        """Test that temperature is clipped to bounds."""
        scheduler = TemperatureScheduler(
            initial_temp=0.07,
            min_temp=0.01,
            max_temp=0.1,
            warmup_steps=10,
            total_steps=100,
            schedule_type="cosine",
        )

        # Run many steps
        for step in range(10000):
            temp = scheduler.step(step)
            assert 0.01 <= temp <= 0.1

    def test_step_without_arg_uses_internal_counter(self):
        """Test that step() without argument uses internal counter."""
        scheduler = TemperatureScheduler(
            initial_temp=0.07,
            schedule_type="constant",
        )

        # Call step() without argument
        scheduler.step()
        assert scheduler.current_step == 1

        scheduler.step()
        assert scheduler.current_step == 2

    def test_get_temperature(self):
        """Test get_temperature method."""
        scheduler = TemperatureScheduler(initial_temp=0.07)

        assert scheduler.get_temperature() == 0.07

        scheduler.step(100)
        assert scheduler.get_temperature() == scheduler.current_temp

    def test_get_state(self):
        """Test get_state method."""
        scheduler = TemperatureScheduler(
            initial_temp=0.07,
            schedule_type="warmup",
        )

        # Run a few steps
        scheduler.step(50, loss=1.5)
        scheduler.step(100, loss=1.3)

        state = scheduler.get_state()

        assert "current_temp" in state
        assert "current_step" in state
        assert "loss_history" in state
        assert "temp_history" in state

        assert state["current_step"] == 100
        assert len(state["loss_history"]) == 2

    def test_load_state(self):
        """Test load_state method."""
        scheduler1 = TemperatureScheduler(
            initial_temp=0.07,
            schedule_type="warmup",
        )

        # Run some steps
        for step in range(50):
            scheduler1.step(step, loss=1.0 + step * 0.01)

        # Save state
        state = scheduler1.get_state()

        # Create new scheduler and load state
        scheduler2 = TemperatureScheduler(
            initial_temp=0.05,  # Different initial value
            schedule_type="constant",  # Different schedule
        )

        scheduler2.load_state(state)

        # Should have same state as scheduler1
        assert scheduler2.current_temp == scheduler1.current_temp
        assert scheduler2.current_step == scheduler1.current_step
        assert scheduler2._loss_history == scheduler1._loss_history

    def test_reset(self):
        """Test reset method."""
        scheduler = TemperatureScheduler(
            initial_temp=0.07,
            schedule_type="warmup",
        )

        # Run some steps
        for step in range(50):
            scheduler.step(step, loss=1.0)

        # Reset
        scheduler.reset()

        # Should be back to initial state
        assert scheduler.current_temp == 0.07
        assert scheduler.current_step == 0
        assert len(scheduler._loss_history) == 0
        assert len(scheduler._temp_history) == 1

    def test_repr(self):
        """Test string representation."""
        scheduler = TemperatureScheduler(
            initial_temp=0.07,
            min_temp=0.01,
            max_temp=0.1,
            schedule_type="warmup",
        )

        repr_str = repr(scheduler)

        assert "TemperatureScheduler" in repr_str
        assert "type=warmup" in repr_str
        assert "current_temp=" in repr_str
        assert "step=" in repr_str
        assert "range=" in repr_str

    def test_different_schedules_produce_different_temps(self):
        """Test that different schedules produce different temperatures."""
        base_params = {
            "initial_temp": 0.07,
            "min_temp": 0.01,
            "max_temp": 0.1,
            "warmup_steps": 100,
            "total_steps": 1000,
        }

        schedulers = {
            "constant": TemperatureScheduler(
                **{**base_params, "schedule_type": "constant"}
            ),
            "warmup": TemperatureScheduler(
                **{**base_params, "schedule_type": "warmup"}
            ),
            "cosine": TemperatureScheduler(
                **{**base_params, "schedule_type": "cosine"}
            ),
            "exponential": TemperatureScheduler(
                **{**base_params, "schedule_type": "exponential"}
            ),
        }

        # Get temperatures at step 500
        temps = {}
        for name, scheduler in schedulers.items():
            if name == "adaptive":
                temps[name] = scheduler.step(500, loss=1.0)
            else:
                temps[name] = scheduler.step(500)

        # Should have different values (at least some)
        unique_temps = len(set(temps.values()))
        assert unique_temps > 1


class TestTemperatureSchedulerEdgeCases:
    """Test edge cases for temperature scheduler."""

    def test_zero_warmup_steps(self):
        """Test with zero warmup steps."""
        scheduler = TemperatureScheduler(
            initial_temp=0.07,
            warmup_steps=0,
            schedule_type="warmup",
        )

        temp = scheduler.step(0)
        assert temp == 0.07  # Should immediately be at initial_temp

    def test_very_small_temperature_range(self):
        """Test with very small temperature range."""
        scheduler = TemperatureScheduler(
            initial_temp=0.07,
            min_temp=0.069,
            max_temp=0.071,
            warmup_steps=100,
            total_steps=1000,
            schedule_type="cosine",
        )

        # Should still work correctly
        for step in range(0, 1000, 100):
            temp = scheduler.step(step)
            assert 0.069 <= temp <= 0.071

    def test_large_step_numbers(self):
        """Test with very large step numbers."""
        scheduler = TemperatureScheduler(
            initial_temp=0.07,
            min_temp=0.01,
            max_temp=0.1,
            warmup_steps=1000,
            total_steps=100000,
            schedule_type="cosine",
        )

        # Should handle large steps correctly
        temp = scheduler.step(1000000)
        assert 0.01 <= temp <= 0.1
