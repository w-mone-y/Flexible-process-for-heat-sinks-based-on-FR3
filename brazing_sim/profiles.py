"""Shared normalized motion profiles.

Keeping time scaling in one small dependency-free module prevents robot,
conveyor and logistics actors from quietly using different acceleration
rules.  The quintic profile has zero velocity and zero acceleration at both
ends, which removes the visible jerk produced by the former cubic copies.
"""

from __future__ import annotations


def clamp_unit(value: float) -> float:
    """Clamp a scalar to the closed normalized interval [0, 1]."""

    return min(1.0, max(0.0, float(value)))


def quintic_time_scaling(value: float) -> float:
    """Return jerk-limited quintic S-curve progress for normalized time."""

    fraction = clamp_unit(value)
    return fraction**3 * (10.0 - 15.0 * fraction + 6.0 * fraction**2)


__all__ = ["clamp_unit", "quintic_time_scaling"]
