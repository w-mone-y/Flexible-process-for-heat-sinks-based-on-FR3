"""Pure product-coordinate geometry generation and validation."""

from __future__ import annotations

from .models import BrazingPath, FinTarget, ProductConfig

MAX_FINS = 12
MAX_PATHS = 24


def fin_y_positions(product: ProductConfig) -> tuple[float, ...]:
    if product.start_offset_y_m is None:
        centre = 0.5 * (product.fin_count - 1)
        return tuple((index - centre) * product.fin_pitch_m for index in range(product.fin_count))
    return tuple(product.start_offset_y_m + index * product.fin_pitch_m for index in range(product.fin_count))


def generate_geometry(product: ProductConfig) -> tuple[tuple[FinTarget, ...], tuple[BrazingPath, ...]]:
    if product.fin_count > MAX_FINS:
        raise ValueError(f"产品{product.product_id}需要{product.fin_count}片翅片，超过对象池上限{MAX_FINS}")
    if product.fin_count * len(product.brazing_sides) > MAX_PATHS:
        raise ValueError(f"产品{product.product_id}的钎料路径超过对象池上限{MAX_PATHS}")
    if product.fin_pitch_m < product.fin_size_m[1]:
        raise ValueError(f"产品{product.product_id}的翅片节距小于翅片厚度，会发生重叠")
    ys = fin_y_positions(product)
    half_width = 0.5 * product.base_size_m[1]
    half_fin_thickness = 0.5 * product.fin_size_m[1]
    if min(ys) - half_fin_thickness < -half_width or max(ys) + half_fin_thickness > half_width:
        raise ValueError(f"产品{product.product_id}的翅片阵列超出基板Y边界")
    x_start = -0.5 * product.base_size_m[0] + product.path_margin_m
    x_end = 0.5 * product.base_size_m[0] - product.path_margin_m
    if x_start >= x_end:
        raise ValueError(f"产品{product.product_id}的路径边距使有效路径长度不大于0")
    base_top = 0.5 * product.base_size_m[2]
    fin_z = base_top + 0.5 * product.fin_size_m[2]
    fins = tuple(FinTarget(f"fin_{index + 1:02d}", index, (0.0, y, fin_z)) for index, y in enumerate(ys))
    paths: list[BrazingPath] = []
    for fin in fins:
        for side in product.brazing_sides:
            sign = -1.0 if side.value == "left" else 1.0
            y = fin.position[1] + sign * product.bead_offset_m
            if abs(y) + 0.5 * product.path_width_m > half_width + 1.0e-12:
                raise ValueError(f"产品{product.product_id}的路径{fin.fin_id}_{side.value}超出基板Y边界")
            paths.append(
                BrazingPath(
                    path_id=f"slot_{fin.index + 1:02d}_{side.value}_brazing_path",
                    fin_id=fin.fin_id,
                    side=side,
                    start=(x_start, y, base_top),
                    end=(x_end, y, base_top),
                    width_m=product.path_width_m,
                )
            )
    return fins, tuple(paths)


__all__ = ["MAX_FINS", "MAX_PATHS", "fin_y_positions", "generate_geometry"]
