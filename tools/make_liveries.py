#!/usr/bin/env python3
"""Generate the 22 fictional team car sprites, deterministically.

Pure geometry: a top-down open-wheel silhouette built from a fixed polygon
set, filled with each team's base + accent colours, a number roundel and the
team initials. No randomness, no external assets — re-running this script
reproduces the committed PNGs byte-for-byte (same Pillow version).

    python tools/make_liveries.py            # writes apexsim/static/cars/*.png

The grid below MUST stay in lockstep with `GRID_2026` in
apexsim/static/index.html and `CAR_SPRITES` in apexsim/src/server.rs:
same team slugs, same car numbers, same order-agnostic name set.
"""

from PIL import Image, ImageDraw, ImageFont
import pathlib

# team slug: (base, accent, dark shade for the wing edges, initials)
TEAMS = {
    "vortex":   ((210, 31, 60), (255, 255, 255), (120, 12, 28), "VOR"),
    "aurora":   ((242, 118, 46), (27, 42, 94), (150, 62, 14), "AUR"),
    "falcon":   ((22, 50, 108), (210, 31, 60), (9, 26, 62), "FAL"),
    "meridian": ((184, 188, 196), (15, 122, 107), (110, 113, 120), "MER"),
    "titan":    ((15, 122, 77), (184, 217, 74), (7, 70, 43), "TIT"),
    "juniper":  ((122, 182, 72), (20, 20, 20), (66, 104, 34), "JUN"),
    "cobalt":   ((30, 111, 217), (242, 118, 46), (14, 60, 124), "COB"),
    "zenith":   ((33, 168, 176), (255, 255, 255), (14, 84, 89), "ZEN"),
    "everest":  ((154, 160, 168), (210, 31, 60), (86, 90, 96), "EVE"),
    "sable":    ((43, 45, 49), (201, 162, 39), (18, 19, 21), "SAB"),
    "polaris":  ((201, 162, 39), (20, 20, 20), (118, 92, 12), "POL"),
}

# (team, number) — 11 teams x 2 cars, matching GRID_2026 in index.html.
GRID = [
    ("vortex", 8), ("vortex", 78),
    ("aurora", 34), ("aurora", 66),
    ("falcon", 51), ("falcon", 52),
    ("meridian", 15), ("meridian", 45),
    ("titan", 64), ("titan", 65),
    ("juniper", 58), ("juniper", 96),
    ("cobalt", 72), ("cobalt", 92),
    ("zenith", 26), ("zenith", 84),
    ("everest", 36), ("everest", 83),
    ("sable", 41), ("sable", 93),
    ("polaris", 19), ("polaris", 95),
]

SIZE = 160          # sprite canvas, px (the dashboard scales on the map)
CAR_LEN = 140       # nose -> rear wing, px (car points UP = towards -y)
CAR_WID = 54        # widest body point, px


def rounded(draw, box, r, fill):
    draw.rounded_rectangle(box, radius=r, fill=fill)


def draw_car(num: int, base, accent, dark, initials: str) -> Image.Image:
    img = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    cx = SIZE // 2
    top = (SIZE - CAR_LEN) // 2          # y of the nose
    bot = top + CAR_LEN                  # y of the rear wing

    # Tyres first (under the body): four black rounded rectangles.
    tyre = (16, 16, 18, 255)
    tyre_w, tyre_h = 14, 30
    for sy in (top + 26, bot - 64):
        for sx in (cx - CAR_WID // 2 - 6, cx + CAR_WID // 2 - 8):
            rounded(d, (sx, sy, sx + tyre_w, sy + tyre_h), 4, tyre)

    # Front wing (wide, thin) and rear wing (wide, thick), dark shade.
    rounded(d, (cx - CAR_WID // 2 - 4, top - 2, cx + CAR_WID // 2 + 4, top + 8), 3, dark + (255,))
    rounded(d, (cx - CAR_WID // 2 - 2, bot - 16, cx + CAR_WID // 2 + 2, bot), 3, dark + (255,))

    # Body: nose -> cockpit bulge -> sidepods -> engine cover, one polygon.
    nose_w, pod_w = 14, CAR_WID
    body = [
        (cx - nose_w // 2, top + 6), (cx + nose_w // 2, top + 6),          # nose tip
        (cx + 12, top + 34),                                               # cockpit right
        (cx + pod_w // 2, top + 46), (cx + pod_w // 2, bot - 24),          # right sidepod
        (cx + 10, bot - 18),                                               # taper right
        (cx - 10, bot - 18),                                               # taper left
        (cx - pod_w // 2, bot - 24), (cx - pod_w // 2, top + 46),          # left sidepod
        (cx - 12, top + 34),                                               # cockpit left
    ]
    d.polygon(body, fill=base + (255,))

    # Accent stripe down the centreline + engine-cover panel.
    d.rectangle((cx - 3, top + 10, cx + 3, bot - 20), fill=accent + (255,))
    d.rectangle((cx - 12, bot - 40, cx + 12, bot - 22), fill=dark + (255,))

    # Cockpit: dark ellipse with a halo arc.
    d.ellipse((cx - 9, top + 34, cx + 9, top + 56), fill=(16, 16, 18, 255))

    # Number roundel on the nose, team-dark ring, white face, dark digits.
    r = 13
    cyc = top + 22
    d.ellipse((cx - r, cyc - r, cx + r, cyc + r), fill=(250, 250, 250, 255),
              outline=dark + (255,), width=2)
    f = _font(14)
    d.text((cx, cyc), str(num), font=f, fill=(20, 20, 20, 255), anchor="mm")

    # Initials micro-text on the rear wing.
    f2 = _font(9)
    d.text((cx, bot - 8), initials, font=f2, fill=(240, 240, 240, 255), anchor="mm")
    return img


_FONT = None
def _font(px: int) -> ImageFont.FreeTypeFont:
    global _FONT
    if _FONT is None:
        # Deterministic across machines: Pillow's built-in bitmap fallback.
        try:
            _FONT = ImageFont.load_default(size=px)
        except TypeError:  # Pillow < 10.1: no sized default
            _FONT = ImageFont.load_default()
    return _FONT


def main() -> None:
    out = pathlib.Path(__file__).resolve().parent.parent / "apexsim" / "static" / "cars"
    out.mkdir(parents=True, exist_ok=True)
    for team, num in GRID:
        base, accent, dark, initials = TEAMS[team]
        name = f"{team}_{num}"
        draw_car(num, base, accent, dark, initials).save(out / f"{name}.png")
    print(f"wrote {len(GRID)} sprites to {out}")


if __name__ == "__main__":
    main()
