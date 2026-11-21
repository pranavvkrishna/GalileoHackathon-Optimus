"""Health views for the pitching-airfoil wake task (ORG-MAE-2).

Listed under ``plugins:`` in ``declared/task_health.yaml``. The view provider
decodes the deliverable through ``ctx.load_evaluation_payload()`` and exposes
``continuous_samples``: for each evaluation window, the temporal RMS
fluctuation of the 20 predicted frames about their own window mean (pixel
variance over the forecast leads, averaged over the grid, square root). It
uses only the forecast itself, not the model's separate mean-field output, so
a forecast that is frozen in time (persistence, a steady or blurred static
field) gives 0 whatever the model claims as its mean. The generic
``sample_dispersion_floor`` check reads it to catch forecasts whose unsteadiness
collapsed or does not vary with the kinematics.
"""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np

from execute_tools.health_checks import (
    CONTINUOUS_SAMPLES,
    ContinuousSamplesPayload,
    HealthView,
    register_view_provider,
)
from execute_tools.health_checks.schemas import HealthCheckContext

PROVIDER_ID = "pitching_wake.prediction_views"


class WakePredictionViews:
    provider_id: ClassVar[str] = PROVIDER_ID
    capabilities: ClassVar[frozenset[str]] = frozenset({CONTINUOUS_SAMPLES})

    def materialize(self, capability_key: str, ctx: HealthCheckContext,
                    config: dict[str, Any] | None = None) -> HealthView:
        decoded = ctx.load_evaluation_payload()
        if not isinstance(decoded, dict) or "forecast" not in decoded:
            raise TypeError(f"expected the decoded prediction arrays, got {type(decoded).__name__}")
        if capability_key != CONTINUOUS_SAMPLES:
            raise KeyError(capability_key)
        forecast = np.asarray(decoded["forecast"], dtype=np.float64)
        rms = np.sqrt(np.mean(forecast.var(axis=1), axis=(1, 2)))
        rms = np.where(np.isfinite(rms), rms, 0.0)
        return HealthView(capability_key=capability_key, provider_id=self.provider_id,
                          payload=ContinuousSamplesPayload(samples=rms.astype(np.float32)))


register_view_provider(WakePredictionViews())
