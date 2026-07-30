"""Generate the V2 worktable plaques used by the MuJoCo scene.

The files are generated rather than hand-edited so station wording, colours,
resolution, and visual hierarchy remain reproducible.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "assets" / "signs"
FONT_CANDIDATES = (
    Path("/System/Library/Fonts/STHeiti Medium.ttc"),
    Path("/System/Library/Fonts/PingFang.ttc"),
    Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
)


@dataclass(frozen=True, slots=True)
class Plaque:
    filename: str
    code: str
    title: str
    accent: str


PLAQUES = (
    Plaque("v2_s1_base_loading_sign.png", "S1", "基板上料", "#6DA8D9"),
    Plaque("v2_s2a_dispensing_sign.png", "S2A", "钎料涂覆", "#D9A05E"),
    Plaque("v2_s2b_coating_inspection_sign.png", "S2B", "焊料检测", "#68B88C"),
    Plaque("v2_fin_a_supply_sign.png", "A线", "翅片上料", "#A97ED1"),
    Plaque("v2_s3a_install_sign.png", "S3A", "翅片装配", "#A97ED1"),
    Plaque("v2_fin_b_supply_sign.png", "B线", "翅片上料", "#72A7D6"),
    Plaque("v2_s3b_install_sign.png", "S3B", "翅片装配", "#72A7D6"),
    Plaque("v2_s4_pre_braze_inspection_sign.png", "S4", "焊前检测", "#68B88C"),
)


def _font(size: int) -> ImageFont.FreeTypeFont:
    for candidate in FONT_CANDIDATES:
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size=size)
    raise FileNotFoundError("未找到支持中文的铭牌字体")


def _centered_y(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.FreeTypeFont,
    height: int,
) -> float:
    bounds = draw.textbbox((0, 0), text, font=font)
    return 0.5 * (height - (bounds[3] - bounds[1])) - bounds[1]


def render(plaque: Plaque) -> Image.Image:
    width, height = 1024, 256
    image = Image.new("RGB", (width, height), "#18232D")
    draw = ImageDraw.Draw(image)
    accent = plaque.accent
    draw.rounded_rectangle(
        (10, 10, width - 11, height - 11),
        radius=34,
        outline=accent,
        width=12,
    )
    draw.rounded_rectangle(
        (40, 38, 252, height - 39),
        radius=26,
        fill=accent,
    )
    code_font = _font(72)
    title_font = _font(92)
    code_bounds = draw.textbbox((0, 0), plaque.code, font=code_font)
    code_x = 146 - 0.5 * (code_bounds[2] - code_bounds[0]) - code_bounds[0]
    draw.text(
        (code_x, _centered_y(draw, plaque.code, code_font, height)),
        plaque.code,
        font=code_font,
        fill="#10202B",
    )
    title_bounds = draw.textbbox((0, 0), plaque.title, font=title_font)
    title_x = 632 - 0.5 * (title_bounds[2] - title_bounds[0]) - title_bounds[0]
    draw.text(
        (title_x, _centered_y(draw, plaque.title, title_font, height)),
        plaque.title,
        font=title_font,
        fill="#F4F7FA",
    )
    draw.ellipse((952, 106, 986, 140), fill=accent)
    return image


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for plaque in PLAQUES:
        render(plaque).save(OUTPUT_DIR / plaque.filename, optimize=True)


if __name__ == "__main__":
    main()
