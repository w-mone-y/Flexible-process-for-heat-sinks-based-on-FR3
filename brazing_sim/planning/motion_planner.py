"""Deterministic PRM/A*, RRT-Connect fallback and space-time reservations.

The planner is independent from MuJoCo.  Callers provide collision and
occupancy callbacks built from the current model, tools, carried workpieces
and turntable sweep.  This keeps planning testable while still allowing the
physical execution layer to use link capsules and exact scene geometry.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from heapq import heappop, heappush
from math import ceil, inf
from typing import Callable, Iterable, Sequence

import numpy as np

JointVector = tuple[float, ...]
CollisionFree = Callable[[np.ndarray], bool]
EdgeFree = Callable[[np.ndarray, np.ndarray], bool]
OccupancyCells = Callable[[np.ndarray], Iterable[str]]


@dataclass(frozen=True, slots=True)
class MotionRequest:
    request_id: str
    resource_id: str
    start: JointVector
    goals: tuple[JointVector, ...]
    earliest_start: float
    max_joint_speed: float = 1.2
    planning_timeout_s: float = 0.35
    tool_id: str | None = None
    carried_object: str | None = None
    lock_joint_indices: tuple[int, ...] = ()
    required_safe_nodes: tuple[str, ...] = ()
    minimum_clearance_m: float = 0.04
    seed: int = 0

    def __post_init__(self) -> None:
        if not self.request_id or not self.resource_id or not self.goals:
            raise ValueError("motion request id, resource and goals are required")
        dimension = len(self.start)
        if dimension == 0 or any(len(goal) != dimension for goal in self.goals):
            raise ValueError("all joint vectors must have the same positive dimension")
        if self.max_joint_speed <= 0 or self.minimum_clearance_m < 0:
            raise ValueError("motion limits must be positive")


@dataclass(frozen=True, slots=True)
class TimedJointSample:
    time: float
    position: JointVector
    safe_wait: bool = False


@dataclass(slots=True)
class JointPath:
    request_id: str
    resource_id: str
    samples: list[TimedJointSample]
    planner: str
    geometric_length: float
    reservation_id: str | None = None
    waiting_time: float = 0.0

    @property
    def start_time(self) -> float:
        return self.samples[0].time

    @property
    def end_time(self) -> float:
        return self.samples[-1].time

    def as_dict(self) -> dict[str, object]:
        return {
            "request_id": self.request_id,
            "resource_id": self.resource_id,
            "planner": self.planner,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration": self.end_time - self.start_time,
            "geometric_length": self.geometric_length,
            "waiting_time": self.waiting_time,
            "reservation_id": self.reservation_id,
            "sample_count": len(self.samples),
        }


@dataclass(frozen=True, slots=True)
class SpaceTimeReservation:
    reservation_id: str
    resource_id: str
    request_id: str
    start_time: float
    end_time: float
    occupied_cells: tuple[tuple[str, float, float], ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "reservation_id": self.reservation_id,
            "resource_id": self.resource_id,
            "request_id": self.request_id,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "occupied_cells": [list(value) for value in self.occupied_cells],
        }


class SpaceTimeReservationTable:
    """SIPP-style interval table for swept link/tool/workpiece cells."""

    def __init__(self, safety_time_s: float = 0.02) -> None:
        self.safety_time_s = float(safety_time_s)
        self._reservations: dict[str, SpaceTimeReservation] = {}
        self._sequence = 0

    @property
    def reservations(self) -> tuple[SpaceTimeReservation, ...]:
        return tuple(self._reservations[key] for key in sorted(self._reservations))

    def release(self, reservation_id: str) -> bool:
        return self._reservations.pop(str(reservation_id), None) is not None

    def release_resource(self, resource_id: str) -> int:
        resource_id = str(resource_id).upper()
        keys = [key for key, value in self._reservations.items() if value.resource_id == resource_id]
        for key in keys:
            del self._reservations[key]
        return len(keys)

    def clear(self) -> None:
        self._reservations.clear()

    @staticmethod
    def _overlaps(first: tuple[float, float], second: tuple[float, float], margin: float) -> bool:
        return first[0] < second[1] + margin and second[0] < first[1] + margin

    def first_conflict(
        self,
        cells: Iterable[tuple[str, float, float]],
        *,
        resource_id: str,
    ) -> tuple[SpaceTimeReservation, float] | None:
        own_resource = str(resource_id).upper()
        for cell, start, end in cells:
            for reservation in self._reservations.values():
                if reservation.resource_id == own_resource:
                    continue
                for other_cell, other_start, other_end in reservation.occupied_cells:
                    if cell == other_cell and self._overlaps(
                        (start, end), (other_start, other_end), self.safety_time_s
                    ):
                        return reservation, other_end + self.safety_time_s
        return None

    def reserve(
        self,
        path: JointPath,
        occupancy: OccupancyCells,
        *,
        sample_period_s: float = 0.02,
        safe_wait: Callable[[np.ndarray], bool] | None = None,
        max_wait_s: float = 30.0,
    ) -> SpaceTimeReservation:
        """Reserve a path, waiting at its certified start node when necessary."""

        cells = self._sample_cells(path, occupancy, sample_period_s)
        total_wait = 0.0
        while True:
            conflict = self.first_conflict(cells, resource_id=path.resource_id)
            if conflict is None:
                break
            if safe_wait is not None and not safe_wait(np.asarray(path.samples[0].position, dtype=float)):
                raise RuntimeError("时空冲突且当前节点不是认证安全等待点")
            wait_until = conflict[1]
            shift = max(self.safety_time_s, wait_until - path.start_time)
            total_wait += shift
            if total_wait > max_wait_s:
                raise TimeoutError("时空路径预约等待超过上限")
            path.samples = [
                TimedJointSample(item.time + shift, item.position, item.safe_wait) for item in path.samples
            ]
            cells = [(cell, start + shift, end + shift) for cell, start, end in cells]
        self._sequence += 1
        reservation_id = f"RES-{self._sequence:06d}-{path.request_id}"
        reservation = SpaceTimeReservation(
            reservation_id=reservation_id,
            resource_id=path.resource_id.upper(),
            request_id=path.request_id,
            start_time=path.start_time,
            end_time=path.end_time,
            occupied_cells=tuple(cells),
        )
        self._reservations[reservation_id] = reservation
        path.reservation_id = reservation_id
        path.waiting_time = total_wait
        return reservation

    @staticmethod
    def _sample_cells(
        path: JointPath,
        occupancy: OccupancyCells,
        sample_period_s: float,
    ) -> list[tuple[str, float, float]]:
        period = min(0.02, max(0.001, float(sample_period_s)))
        sampled: list[tuple[str, float, float]] = []
        for first, second in zip(path.samples, path.samples[1:]):
            duration = max(period, second.time - first.time)
            count = max(1, int(ceil(duration / period)))
            q0 = np.asarray(first.position, dtype=float)
            q1 = np.asarray(second.position, dtype=float)
            for index in range(count):
                fraction = index / count
                time = first.time + fraction * duration
                q = q0 + fraction * (q1 - q0)
                for cell in occupancy(q):
                    sampled.append((str(cell), time, min(second.time, time + period)))
        return sampled


@dataclass(slots=True)
class ProbabilisticRoadmap:
    dimension: int
    nodes: list[np.ndarray] = field(default_factory=list)
    edges: dict[int, list[tuple[int, float]]] = field(default_factory=dict)

    def add_node(self, value: Sequence[float]) -> int:
        vector = np.asarray(value, dtype=float)
        if vector.shape != (self.dimension,):
            raise ValueError("roadmap node has incorrect dimension")
        index = len(self.nodes)
        self.nodes.append(vector)
        self.edges[index] = []
        return index

    def connect(self, first: int, second: int) -> None:
        distance = float(np.linalg.norm(self.nodes[first] - self.nodes[second]))
        if not any(index == second for index, _ in self.edges[first]):
            self.edges[first].append((second, distance))
            self.edges[second].append((first, distance))


class HybridMotionPlanner:
    """Joint-space planner with deterministic static roadmap and online fallback."""

    def __init__(
        self,
        joint_limits: Sequence[tuple[float, float]],
        collision_free: CollisionFree,
        edge_free: EdgeFree | None = None,
        *,
        seed: int = 0,
        roadmap_samples: int = 180,
        neighbours: int = 10,
        edge_resolution: float = 0.08,
    ) -> None:
        self.joint_limits = tuple((float(low), float(high)) for low, high in joint_limits)
        if not self.joint_limits or any(low >= high for low, high in self.joint_limits):
            raise ValueError("invalid joint limits")
        self.dimension = len(self.joint_limits)
        self.collision_free = collision_free
        self.edge_resolution = float(edge_resolution)
        self.edge_free = edge_free or self._sampled_edge_free
        self.seed = int(seed)
        self.neighbours = int(neighbours)
        self.roadmap = ProbabilisticRoadmap(self.dimension)
        self._build_roadmap(int(roadmap_samples))

    def _sampled_edge_free(self, first: np.ndarray, second: np.ndarray) -> bool:
        distance = float(np.linalg.norm(second - first))
        count = max(1, int(ceil(distance / max(self.edge_resolution, 1e-4))))
        return all(
            self.collision_free(first + (index / count) * (second - first)) for index in range(count + 1)
        )

    def _build_roadmap(self, sample_count: int) -> None:
        rng = np.random.default_rng(self.seed)
        attempts = 0
        while len(self.roadmap.nodes) < sample_count and attempts < sample_count * 30:
            attempts += 1
            sample = np.asarray([rng.uniform(low, high) for low, high in self.joint_limits])
            if self.collision_free(sample):
                self.roadmap.add_node(sample)
        for index, node in enumerate(self.roadmap.nodes):
            neighbours = sorted(
                (
                    (float(np.linalg.norm(node - other)), other_index)
                    for other_index, other in enumerate(self.roadmap.nodes)
                    if other_index != index
                ),
                key=lambda item: (item[0], item[1]),
            )[: self.neighbours]
            for _, other_index in neighbours:
                if self.edge_free(node, self.roadmap.nodes[other_index]):
                    self.roadmap.connect(index, other_index)

    def _nearest_visible(self, vector: np.ndarray) -> list[tuple[int, float]]:
        candidates = sorted(
            ((index, float(np.linalg.norm(vector - node))) for index, node in enumerate(self.roadmap.nodes)),
            key=lambda item: (item[1], item[0]),
        )
        return [
            item
            for item in candidates[: self.neighbours]
            if self.edge_free(vector, self.roadmap.nodes[item[0]])
        ]

    def _astar(self, start: np.ndarray, goal: np.ndarray) -> list[np.ndarray] | None:
        start_edges = self._nearest_visible(start)
        goal_edges = self._nearest_visible(goal)
        if self.edge_free(start, goal):
            return [start, goal]
        if not start_edges or not goal_edges:
            return None
        goal_cost = {index: distance for index, distance in goal_edges}
        queue: list[tuple[float, float, int]] = []
        costs: dict[int, float] = {}
        parents: dict[int, int] = {}
        for index, distance in start_edges:
            costs[index] = distance
            heuristic = float(np.linalg.norm(self.roadmap.nodes[index] - goal))
            heappush(queue, (distance + heuristic, distance, index))
        reached: int | None = None
        while queue:
            _, cost, index = heappop(queue)
            if cost > costs.get(index, inf) + 1e-12:
                continue
            if index in goal_cost:
                reached = index
                break
            for neighbour, edge_cost in self.roadmap.edges[index]:
                candidate = cost + edge_cost
                if candidate + 1e-12 >= costs.get(neighbour, inf):
                    continue
                costs[neighbour] = candidate
                parents[neighbour] = index
                heuristic = float(np.linalg.norm(self.roadmap.nodes[neighbour] - goal))
                heappush(queue, (candidate + heuristic, candidate, neighbour))
        if reached is None:
            return None
        indices = [reached]
        while indices[-1] in parents:
            indices.append(parents[indices[-1]])
        indices.reverse()
        return [start, *(self.roadmap.nodes[index] for index in indices), goal]

    def _rrt_connect(self, start: np.ndarray, goal: np.ndarray, seed: int) -> list[np.ndarray] | None:
        rng = np.random.default_rng(seed)
        first = [start]
        second = [goal]
        first_parents = [-1]
        second_parents = [-1]
        step = 0.22

        def extend(tree: list[np.ndarray], parents: list[int], target: np.ndarray) -> int | None:
            nearest = min(range(len(tree)), key=lambda index: float(np.linalg.norm(tree[index] - target)))
            delta = target - tree[nearest]
            distance = float(np.linalg.norm(delta))
            candidate = target if distance <= step else tree[nearest] + delta * (step / distance)
            if not self.edge_free(tree[nearest], candidate):
                return None
            tree.append(candidate)
            parents.append(nearest)
            return len(tree) - 1

        for iteration in range(1800):
            sample = (
                goal
                if iteration % 8 == 0
                else np.asarray([rng.uniform(low, high) for low, high in self.joint_limits])
            )
            added = extend(first, first_parents, sample)
            if added is None:
                continue
            connected = extend(second, second_parents, first[added])
            if connected is not None and self.edge_free(first[added], second[connected]):

                def trace(tree: list[np.ndarray], parents: list[int], index: int) -> list[np.ndarray]:
                    result = [tree[index]]
                    while parents[index] >= 0:
                        index = parents[index]
                        result.append(tree[index])
                    result.reverse()
                    return result

                left = trace(first, first_parents, added)
                right = trace(second, second_parents, connected)
                if np.linalg.norm(left[0] - start) > 1e-9:
                    left, right = list(reversed(right)), list(reversed(left))
                return [*left, *reversed(right)]
            first, second = second, first
            first_parents, second_parents = second_parents, first_parents
        return None

    @staticmethod
    def _deduplicate(path: Sequence[np.ndarray]) -> list[np.ndarray]:
        result: list[np.ndarray] = []
        for value in path:
            if not result or np.linalg.norm(value - result[-1]) > 1e-9:
                result.append(value)
        return result

    def plan(self, request: MotionRequest) -> JointPath:
        start = np.asarray(request.start, dtype=float)
        if not self.collision_free(start):
            raise RuntimeError("运动起点处于碰撞状态")
        candidates: list[tuple[float, list[np.ndarray], str]] = []
        for goal_index, goal_value in enumerate(request.goals):
            goal = np.asarray(goal_value, dtype=float)
            if not self.collision_free(goal):
                continue
            path = self._astar(start, goal)
            planner = "PRM_ASTAR"
            if path is None:
                path = self._rrt_connect(start, goal, request.seed + goal_index + self.seed)
                planner = "RRT_CONNECT"
            if path is None:
                continue
            path = self._deduplicate(path)
            length = sum(float(np.linalg.norm(second - first)) for first, second in zip(path, path[1:]))
            candidates.append((length, path, planner))
        if not candidates:
            raise RuntimeError("PRM与RRT-Connect均未找到安全路径")
        length, positions, planner = min(candidates, key=lambda item: (item[0], item[2]))
        samples: list[TimedJointSample] = []
        current_time = float(request.earliest_start)
        samples.append(TimedJointSample(current_time, tuple(float(value) for value in positions[0]), True))
        for first, second in zip(positions, positions[1:]):
            delta = np.abs(second - first)
            duration = float(np.max(delta)) / request.max_joint_speed
            current_time += max(0.001, duration)
            samples.append(TimedJointSample(current_time, tuple(float(value) for value in second), True))
        return JointPath(request.request_id, request.resource_id.upper(), samples, planner, length)


def always_collision_free(_: np.ndarray) -> bool:
    """Convenience predicate used by dry-run planning and unit tests."""

    return True


__all__ = [
    "HybridMotionPlanner",
    "JointPath",
    "MotionRequest",
    "ProbabilisticRoadmap",
    "SpaceTimeReservation",
    "SpaceTimeReservationTable",
    "TimedJointSample",
    "always_collision_free",
]
