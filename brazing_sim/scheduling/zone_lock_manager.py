"""Shared workspace-zone locks with timeout and deadlock diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from threading import RLock
from typing import Any, Iterable


@dataclass(frozen=True, slots=True)
class ZoneLease:
    zone_id: str
    task_id: str
    resource_id: str
    acquired_at: float
    expires_at: float | None = None

    def expired(self, now: float) -> bool:
        return self.expires_at is not None and now >= self.expires_at


class ZoneLockManager:
    def __init__(self, zones: Iterable[str] = (), *, default_timeout: float = 600.0) -> None:
        self.zones = {str(zone).upper() for zone in zones}
        self.default_timeout = float(default_timeout)
        if not isfinite(self.default_timeout) or self.default_timeout <= 0:
            raise ValueError("default_timeout must be finite and positive")
        self._leases: dict[str, ZoneLease] = {}
        self._lock = RLock()
        self.conflict_count = 0

    def register(self, zone_id: str) -> None:
        self.zones.add(str(zone_id).upper())

    def _validate(self, zones: Iterable[str]) -> tuple[str, ...]:
        result = tuple(dict.fromkeys(str(zone).upper() for zone in zones))
        missing = sorted(set(result) - self.zones)
        if missing:
            raise KeyError(f"unknown zones: {missing}")
        return result

    def update(self, now: float) -> list[ZoneLease]:
        expired = [lease for lease in self._leases.values() if lease.expired(now)]
        for lease in expired:
            self._leases.pop(lease.zone_id, None)
        return expired

    def can_acquire(
        self,
        task_id: str,
        resource_id: str,
        zones: Iterable[str],
        now: float | None = None,
    ) -> bool:
        names = self._validate(zones)
        with self._lock:
            if now is not None:
                self.update(float(now))
            return all(
                zone not in self._leases
                or (
                    self._leases[zone].task_id == task_id
                    and self._leases[zone].resource_id == str(resource_id).upper()
                )
                for zone in names
            )

    def acquire(
        self,
        task_id: str,
        resource_id: str,
        zones: Iterable[str],
        now: float = 0.0,
        timeout: float | None = None,
    ) -> bool:
        names = self._validate(zones)
        resource = str(resource_id).upper()
        ttl = self.default_timeout if timeout is None else float(timeout)
        if not self.can_acquire(task_id, resource, names, now):
            self.conflict_count += 1
            return False
        with self._lock:
            for zone in names:
                self._leases[zone] = ZoneLease(zone, str(task_id), resource, float(now), float(now) + ttl)
        return True

    def release(self, task_id: str) -> int:
        with self._lock:
            names = [zone for zone, lease in self._leases.items() if lease.task_id == task_id]
            for zone in names:
                del self._leases[zone]
            return len(names)

    def release_resource(self, resource_id: str) -> int:
        resource = str(resource_id).upper()
        with self._lock:
            names = [zone for zone, lease in self._leases.items() if lease.resource_id == resource]
            for zone in names:
                del self._leases[zone]
            return len(names)

    def reset(self) -> None:
        self._leases.clear()
        self.conflict_count = 0

    def blockers(self, zones: Iterable[str]) -> dict[str, str]:
        return {zone: self._leases[zone].task_id for zone in self._validate(zones) if zone in self._leases}

    def diagnose(self, now: float) -> dict[str, Any]:
        expired = self.update(now)
        return {
            "expired_leases": [lease.zone_id for lease in expired],
            "active_count": len(self._leases),
            "conflict_count": self.conflict_count,
            "possible_deadlock": len(self._leases) > 1
            and len({lease.task_id for lease in self._leases.values()}) > 1,
        }

    def snapshot(self) -> dict[str, dict[str, Any] | None]:
        return {
            zone: (
                None
                if zone not in self._leases
                else {
                    "task_id": self._leases[zone].task_id,
                    "resource_id": self._leases[zone].resource_id,
                    "acquired_at": self._leases[zone].acquired_at,
                    "expires_at": self._leases[zone].expires_at,
                }
            )
            for zone in sorted(self.zones)
        }


__all__ = ["ZoneLease", "ZoneLockManager"]
