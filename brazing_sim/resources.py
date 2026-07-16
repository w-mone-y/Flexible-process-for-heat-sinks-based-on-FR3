"""Atomic resource leases used by the event-driven coordinator."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import isfinite
from threading import RLock
from typing import Iterable


class ResourceName(str, Enum):
    ASSEMBLY_FIXTURE = "assembly_fixture"
    BRAZING_ZONE = "brazing_zone"
    FURNACE_MOUTH = "furnace_mouth"
    INSPECTION_ZONE = "inspection_zone"

    def __str__(self) -> str:
        return self.value


DEFAULT_RESOURCES = tuple(resource.value for resource in ResourceName)


@dataclass(frozen=True, slots=True)
class ResourceLease:
    resource: str
    owner: str
    acquired_at: float
    expires_at: float | None = None

    def expired(self, now: float) -> bool:
        return self.expires_at is not None and now >= self.expires_at


class ResourceManager:
    """Small in-process lease manager with atomic multi-resource acquisition.

    Re-acquiring a resource by the same owner is idempotent.  A coordinator is
    expected to release a lease only after the actor exits the corresponding
    safety boundary.
    """

    def __init__(self, resources: Iterable[str | ResourceName] = DEFAULT_RESOURCES) -> None:
        names = tuple(dict.fromkeys(str(resource) for resource in resources))
        if not names or any(not name for name in names):
            raise ValueError("at least one named resource is required")
        self._resources = set(names)
        self._leases: dict[str, ResourceLease] = {}
        self._lock = RLock()
        self.conflict_count = 0

    def register(self, resource: str | ResourceName) -> None:
        name = str(resource)
        if not name:
            raise ValueError("resource name must not be empty")
        with self._lock:
            self._resources.add(name)

    def _validate(self, resource: str | ResourceName) -> str:
        name = str(resource)
        if name not in self._resources:
            raise KeyError(f"unknown resource: {name}")
        return name

    @staticmethod
    def _expiry(now: float, ttl: float | None) -> float | None:
        if not isfinite(now):
            raise ValueError("now must be finite")
        if ttl is None:
            return None
        if not isfinite(ttl) or ttl <= 0:
            raise ValueError("ttl must be finite and positive")
        return now + ttl

    def _expire_locked(self, now: float) -> None:
        expired = [resource for resource, lease in self._leases.items() if lease.expired(now)]
        for resource in expired:
            del self._leases[resource]

    def acquire(
        self,
        resource: str | ResourceName,
        owner: str,
        now: float = 0.0,
        ttl: float | None = None,
    ) -> bool:
        return self.acquire_many((resource,), owner, now=now, ttl=ttl)

    def acquire_many(
        self,
        resources: Iterable[str | ResourceName],
        owner: str,
        now: float = 0.0,
        ttl: float | None = None,
    ) -> bool:
        if not owner:
            raise ValueError("lease owner must not be empty")
        names = tuple(dict.fromkeys(self._validate(resource) for resource in resources))
        if not names:
            return True
        expires_at = self._expiry(now, ttl)
        with self._lock:
            self._expire_locked(now)
            blockers = [name for name in names if name in self._leases and self._leases[name].owner != owner]
            if blockers:
                self.conflict_count += 1
                return False
            for name in names:
                existing = self._leases.get(name)
                if existing is None:
                    self._leases[name] = ResourceLease(name, owner, now, expires_at)
                elif ttl is not None:
                    # An explicit TTL refreshes an owner's existing lease.
                    self._leases[name] = ResourceLease(name, owner, existing.acquired_at, expires_at)
            return True

    def release(self, resource: str | ResourceName, owner: str) -> bool:
        name = self._validate(resource)
        with self._lock:
            lease = self._leases.get(name)
            if lease is None or lease.owner != owner:
                return False
            del self._leases[name]
            return True

    def release_all(self, owner: str | None = None) -> int:
        """Release all leases, or only those held by ``owner``."""

        with self._lock:
            names = [
                resource for resource, lease in self._leases.items() if owner is None or lease.owner == owner
            ]
            for resource in names:
                del self._leases[resource]
            return len(names)

    def reset(self) -> None:
        """Release every lease and clear per-order conflict accounting."""

        self.release_all()
        self.conflict_count = 0

    def update(self, now: float) -> None:
        if not isfinite(now):
            raise ValueError("now must be finite")
        with self._lock:
            self._expire_locked(now)

    def held_by(self, resource: str | ResourceName, now: float | None = None) -> str | None:
        name = self._validate(resource)
        with self._lock:
            if now is not None:
                self._expire_locked(now)
            lease = self._leases.get(name)
            return None if lease is None else lease.owner

    def owner_resources(self, owner: str, now: float | None = None) -> tuple[str, ...]:
        with self._lock:
            if now is not None:
                self._expire_locked(now)
            return tuple(sorted(name for name, lease in self._leases.items() if lease.owner == owner))

    def snapshot(self, now: float | None = None) -> dict[str, dict[str, float | str | None] | None]:
        with self._lock:
            if now is not None:
                self._expire_locked(now)
            result: dict[str, dict[str, float | str | None] | None] = {}
            for name in sorted(self._resources):
                lease = self._leases.get(name)
                result[name] = (
                    None
                    if lease is None
                    else {
                        "owner": lease.owner,
                        "acquired_at": lease.acquired_at,
                        "expires_at": lease.expires_at,
                    }
                )
            return result

    @property
    def leases(self) -> tuple[ResourceLease, ...]:
        with self._lock:
            return tuple(self._leases[name] for name in sorted(self._leases))
