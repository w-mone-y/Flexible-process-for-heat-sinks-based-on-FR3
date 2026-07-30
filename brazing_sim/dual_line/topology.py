"""Authoritative station graph for the independent V2 production line."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from types import MappingProxyType
from typing import Iterable, Mapping


@dataclass(frozen=True, slots=True)
class Station:
    station_id: str
    label_zh: str
    world_xyz: tuple[float, float, float]
    capacity: int = 1

    def __post_init__(self) -> None:
        if not self.station_id or self.capacity < 1:
            raise ValueError("station id must be non-empty and capacity must be positive")


class DualLineTopology:
    """Small immutable interface hiding all V2 coordinates and route edges."""

    def __init__(
        self,
        stations: Iterable[Station],
        edges: Mapping[str, Iterable[str]],
    ) -> None:
        station_map = {station.station_id: station for station in stations}
        if len(station_map) == 0:
            raise ValueError("topology requires stations")
        self._stations = MappingProxyType(station_map)
        self._edges = MappingProxyType(
            {station_id: tuple(successors) for station_id, successors in edges.items()}
        )

    @classmethod
    def standard(cls) -> "DualLineTopology":
        stations = (
            Station("S1_BASE_LOADING", "S1 基板上料", (-0.55, 0.35, 0.225)),
            Station("S2A_DISPENSING", "S2A 钎料涂覆", (-0.35, -0.10, 0.225)),
            Station("S2B_MATERIAL_INSPECTION", "S2B 焊料检测与分流", (0.50, 0.00, 0.225)),
            Station("S3A_ARM1_INSTALL", "S3A Arm1 翅片安装", (0.55, 0.50, 0.225)),
            Station("FIN_TABLE_A", "A线翅片料台", (0.45, 1.05, 0.265)),
            Station("S3B_ARM3_INSTALL", "S3B Arm3 翅片安装", (0.35, -0.45, 0.225)),
            Station("FIN_TABLE_B", "B线翅片料台", (0.55, -0.85, 0.265)),
            Station("MERGE_A_WAIT", "A线北侧平面等待位", (1.40, 0.50, 0.225)),
            Station("MERGE_B_WAIT", "B线南侧平面等待位", (1.55, -1.22, 0.225)),
            Station("Y_MERGE_SHARED", "S4入口单占用区", (1.40, 0.00, 0.225)),
            Station("S4_PRE_BRAZE_INSPECTION", "S4 共享焊前检测", (1.40, 0.00, 0.225)),
            Station("FURNACE_BUFFER_1", "炉前缓存1", (1.85, 0.00, 0.225)),
            Station("FURNACE_BUFFER_2", "炉前缓存2", (2.30, 0.00, 0.225)),
            Station("FURNACE_BUFFER_3", "炉前缓存3", (2.75, 0.00, 0.225)),
            Station("FURNACE_FRONT", "贯通炉前门", (3.12, 0.00, 0.225)),
            Station("FURNACE_LAYER_0", "炉内底层", (3.45, 0.00, 0.225)),
            Station("FURNACE_LAYER_1", "炉内中层", (3.45, 0.00, 0.365)),
            Station("FURNACE_LAYER_2", "炉内顶层", (3.45, 0.00, 0.505)),
            Station("FURNACE_REAR", "贯通炉后门", (3.78, 0.00, 0.225)),
            Station("POST_BRAZE_SCAN", "固定焊后视觉门架", (4.20, 0.00, 0.225)),
            Station("FINISHED_OUTPUT", "成品出口", (4.92, 0.00, 0.225)),
        )
        edges = {
            "S1_BASE_LOADING": ("S2A_DISPENSING",),
            "S2A_DISPENSING": ("S2B_MATERIAL_INSPECTION",),
            "S2B_MATERIAL_INSPECTION": ("S3A_ARM1_INSTALL", "S3B_ARM3_INSTALL"),
            "S3A_ARM1_INSTALL": ("MERGE_A_WAIT",),
            "S3B_ARM3_INSTALL": ("MERGE_B_WAIT",),
            "MERGE_A_WAIT": ("Y_MERGE_SHARED",),
            "MERGE_B_WAIT": ("Y_MERGE_SHARED",),
            "Y_MERGE_SHARED": ("S4_PRE_BRAZE_INSPECTION",),
            "S4_PRE_BRAZE_INSPECTION": ("FURNACE_BUFFER_1",),
            "FURNACE_BUFFER_1": ("FURNACE_BUFFER_2", "FURNACE_FRONT"),
            "FURNACE_BUFFER_2": ("FURNACE_BUFFER_3", "FURNACE_FRONT"),
            "FURNACE_BUFFER_3": ("FURNACE_FRONT",),
            "FURNACE_FRONT": ("FURNACE_LAYER_0", "FURNACE_LAYER_1", "FURNACE_LAYER_2"),
            "FURNACE_LAYER_0": ("FURNACE_REAR",),
            "FURNACE_LAYER_1": ("FURNACE_REAR",),
            "FURNACE_LAYER_2": ("FURNACE_REAR",),
            "FURNACE_REAR": ("POST_BRAZE_SCAN",),
            "POST_BRAZE_SCAN": ("FINISHED_OUTPUT",),
            "FINISHED_OUTPUT": (),
            "FIN_TABLE_A": (),
            "FIN_TABLE_B": (),
        }
        return cls(stations, edges)

    _OWNER_STATIONS = MappingProxyType(
        {
            "S1": "S1_BASE_LOADING",
            "S2A": "S2A_DISPENSING",
            "S2B": "S2B_MATERIAL_INSPECTION",
            "INSTALL_A": "S3A_ARM1_INSTALL",
            "INSTALL_B": "S3B_ARM3_INSTALL",
            "MERGE_A_WAIT": "MERGE_A_WAIT",
            "MERGE_B_WAIT": "MERGE_B_WAIT",
            "MERGE": "Y_MERGE_SHARED",
            "S4": "S4_PRE_BRAZE_INSPECTION",
            "BUFFER_1": "FURNACE_BUFFER_1",
            "BUFFER_2": "FURNACE_BUFFER_2",
            "BUFFER_3": "FURNACE_BUFFER_3",
            "POST_SCAN": "POST_BRAZE_SCAN",
            "OUTPUT": "FINISHED_OUTPUT",
        }
    )

    @property
    def stations(self) -> tuple[Station, ...]:
        return tuple(self._stations.values())

    def station(self, station_id: str) -> Station:
        try:
            return self._stations[str(station_id)]
        except KeyError as exc:
            raise KeyError(f"unknown V2 station: {station_id}") from exc

    def successors(self, station_id: str) -> tuple[str, ...]:
        self.station(station_id)
        return self._edges.get(str(station_id), ())

    def station_for_owner(self, owner: str) -> Station:
        try:
            station_id = self._OWNER_STATIONS[str(owner)]
        except KeyError as exc:
            raise KeyError(f"V2 owner has no physical station: {owner}") from exc
        return self.station(station_id)

    def route(self, source: str, target: str) -> tuple[str, ...]:
        self.station(source)
        self.station(target)
        queue: deque[tuple[str, tuple[str, ...]]] = deque([(source, (source,))])
        visited = {source}
        while queue:
            current, path = queue.popleft()
            if current == target:
                return path
            for successor in self.successors(current):
                if successor not in visited:
                    visited.add(successor)
                    queue.append((successor, (*path, successor)))
        raise ValueError(f"no V2 route from {source} to {target}")

    def validate(self) -> tuple[str, ...]:
        errors: list[str] = []
        for station_id, successors in self._edges.items():
            if station_id not in self._stations:
                errors.append(f"edge source is missing: {station_id}")
            for successor in successors:
                if successor not in self._stations:
                    errors.append(f"edge target is missing: {successor}")
        required = {
            "S1_BASE_LOADING",
            "S2A_DISPENSING",
            "S2B_MATERIAL_INSPECTION",
            "S3A_ARM1_INSTALL",
            "S3B_ARM3_INSTALL",
            "S4_PRE_BRAZE_INSPECTION",
            "FURNACE_FRONT",
            "FURNACE_REAR",
            "POST_BRAZE_SCAN",
            "FINISHED_OUTPUT",
        }
        missing = sorted(required.difference(self._stations))
        errors.extend(f"required station is missing: {station_id}" for station_id in missing)
        if not errors:
            try:
                self.route("S1_BASE_LOADING", "FINISHED_OUTPUT")
            except ValueError as exc:
                errors.append(str(exc))
        return tuple(errors)

    def as_dict(self) -> dict[str, object]:
        return {
            "stations": [
                {
                    "station_id": station.station_id,
                    "label_zh": station.label_zh,
                    "world_xyz": list(station.world_xyz),
                    "capacity": station.capacity,
                    "successors": list(self.successors(station.station_id)),
                }
                for station in self.stations
            ]
        }


__all__ = ["DualLineTopology", "Station"]
