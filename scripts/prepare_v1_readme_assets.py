"""Polish the historical V1 snapshots into a consistent README storyboard.

V1 is intentionally documented as a historical single-line reference.  This
script only normalises presentation (size, labels and transitions); it does
not claim that the stills are a continuous physical replay.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "docs" / "images" / "readme"
WIDTH, HEIGHT = 1280, 720
FONT_PATH = "/System/Library/Fonts/Hiragino Sans GB.ttc"

STAGES = (
    ("line_overview.png", "① 单线总览", "V1 固定工艺链"),
    ("material_application.png", "② 钎料涂覆", "Arm2 按固定顺序完成涂覆"),
    ("fin_assembly.png", "③ 翅片安装", "Arm1 逐片安装翅片"),
    ("furnace_cycle.png", "④ 炉内钎焊", "托盘进入炉体完成热循环"),
    ("finished_delivery.png", "⑤ 成品交付", "焊后沿出口送出"),
)


def _font(size: int):
    try:
        return ImageFont.truetype(FONT_PATH, size)
    except OSError:
        return ImageFont.load_default()


def _card(source: Path, title: str, subtitle: str) -> Image.Image:
    image = Image.open(source).convert("RGB").resize((WIDTH, HEIGHT), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (WIDTH, HEIGHT), "#0f172a")
    canvas.paste(image, (0, 0))
    draw = ImageDraw.Draw(canvas, "RGBA")
    draw.rectangle((0, 0, WIDTH, 104), fill=(15, 23, 42, 230))
    draw.rectangle((0, HEIGHT - 74, WIDTH, HEIGHT), fill=(15, 23, 42, 220))
    draw.text((34, 22), title, fill="#f8fafc", font=_font(42))
    draw.text((36, HEIGHT - 58), subtitle, fill="#cbd5e1", font=_font(28))
    draw.rounded_rectangle((WIDTH - 178, 22, WIDTH - 30, 78), radius=14, fill="#2563eb")
    draw.text((WIDTH - 146, 32), "V1", fill="white", font=_font(32))
    return canvas


def main() -> int:
    SOURCE.mkdir(parents=True, exist_ok=True)
    cards = []
    for index, (filename, title, subtitle) in enumerate(STAGES, start=1):
        card = _card(SOURCE / filename, title, subtitle)
        card.save(SOURCE / f"v1_stage_{index}.png", optimize=True)
        cards.append(card)

    cards[0].save(
        SOURCE / "v1_process_tour.webp",
        save_all=True,
        append_images=cards[1:],
        duration=1100,
        loop=0,
        format="WEBP",
        quality=94,
        method=6,
    )
    cards[0].save(
        SOURCE / "v1_process_tour.gif",
        save_all=True,
        append_images=cards[1:],
        duration=1100,
        loop=0,
        optimize=True,
    )
    # A compact contact sheet is useful when README is viewed on mobile.
    sheet = Image.new("RGB", (1280, 800), "#020617")
    thumb_w, thumb_h = 400, 225
    for index, card in enumerate(cards):
        thumb = card.resize((thumb_w, thumb_h), Image.Resampling.LANCZOS)
        x = 28 + (index % 3) * 416
        y = 28 + (index // 3) * 366
        sheet.paste(thumb, (x, y))
        ImageDraw.Draw(sheet).text((x, y + thumb_h + 12), STAGES[index][1], fill="#e2e8f0", font=_font(24))
    sheet.save(SOURCE / "v1_process_storyboard.png", optimize=True)
    print(f"generated {len(cards)} V1 stage cards and tour animations")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
