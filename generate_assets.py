"""Meta Horizon Store artwork for Vista Pictures VR.

The look is QuestBeats': near-black ground, a faint square grid that fades out towards the
edges, three soft radial glows, the app icon with a halo, wordmark, caption. Kept deliberately,
so the three apps read as one developer's shelf. What changes is the palette — Vista's own
purple and cyan, taken from the app theme and from the icon itself — and the wordmark.

Two departures from the QuestBeats scripts, both about time rather than taste:

  - The glows and the grid mask are computed with numpy over the whole array instead of a
    per-pixel Python loop. At 2880x1620 that loop is 4.7 million iterations times three glows;
    the same result comes out of numpy in a few milliseconds, and it is the same arithmetic.
  - The grid fade is a smoothstep rather than a hard linear ramp, which stops the faint banding
    that shows up on the wide canvases where the gradient is stretched furthest.

Usage:
    python generate_assets.py          # writes every size into store_art/
"""

import math
import os

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "store_art")
ICON_SRC = os.path.join(HERE, "..", "app", "src", "main", "res", "mipmap-xxxhdpi", "ic_launcher.png")

# ── Palette ───────────────────────────────────────────────────────────────────
# BG and PURPLE are VrBackground and VrPrimary from the app's Color.kt; CYAN is sampled from
# the sun in the launcher icon, so the artwork and the icon cannot drift apart.
BG = (7, 5, 14)
PURPLE = (155, 107, 255)
CYAN = (45, 212, 235)
PINK = (241, 91, 181)
TEXT = (239, 234, 251)

FONT_BOLD = "C:/Windows/Fonts/seguisb.ttf"      # Segoe UI Semibold
FONT_BLACK = "C:/Windows/Fonts/seguibl.ttf"     # Segoe UI Black
FONT_LIGHT = "C:/Windows/Fonts/segoeuil.ttf"    # Segoe UI Light

WORDMARK = "Vista Pictures VR"
CAPTION = "PHOTOS  ·  3D MODELS  ·  HDRI  ·  TEXTURES"


def font(path, size):
    try:
        return ImageFont.truetype(path, size)
    except OSError:
        return ImageFont.load_default()


def background(W, H):
    """Ground, glows and grid — everything behind the icon."""
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    img = np.zeros((H, W, 3), np.float32)
    img[:] = BG

    # Placed by fraction of the canvas so the same three lights land sensibly on a 3000x900
    # banner and on a 1008x1440 poster. Radius keys off the diagonal for the same reason.
    # Radius keys off the LONG edge, not the diagonal, and the strengths are low on purpose.
    # The ground has to stay near-black: the grid is only visible against black, and a glow
    # strong enough to tint the whole canvas washes it out completely.
    glows = [
        (0.14, -0.06, 0.55, PURPLE, 0.20),
        (0.94, 0.16, 0.50, CYAN, 0.15),
        (0.52, 1.10, 0.48, PINK, 0.10),
    ]
    for fx, fy, fr, colour, strength in glows:
        cx, cy, r = fx * W, fy * H, fr * max(W, H)
        d2 = ((xx - cx) ** 2 + (yy - cy) ** 2) / (r * r)
        a = np.clip(1.0 - d2, 0.0, 1.0) ** 2 * strength
        img += (np.array(colour, np.float32) - img) * a[..., None]

    base = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))

    # Grid: drawn opaque, then faded by distance from the centre so the edges go quiet.
    step = max(48, min(W, H) // 14)
    grid = Image.new("L", (W, H), 0)
    gd = ImageDraw.Draw(grid)
    for x in range(0, W + step, step):
        gd.line([(x, 0), (x, H)], fill=255, width=1)
    for y in range(0, H + step, step):
        gd.line([(0, y), (W, y)], fill=255, width=1)

    d = np.sqrt(((xx - W / 2) / (W * 0.62)) ** 2 + ((yy - H / 2) / (H * 0.62)) ** 2)
    t = np.clip(1.0 - d, 0.0, 1.0)
    fade = t * t * (3 - 2 * t)                      # smoothstep, no banding
    alpha = (np.asarray(grid, np.float32) / 255.0) * fade * 15
    grid_rgba = Image.fromarray(
        np.dstack([
            np.full((H, W), 255, np.uint8), np.full((H, W), 255, np.uint8),
            np.full((H, W), 255, np.uint8), alpha.astype(np.uint8),
        ])
    )
    return Image.alpha_composite(base.convert("RGBA"), grid_rgba).convert("RGB")


def icon_tile(size):
    """The launcher icon, cropped to its own tile and given rounded corners plus a halo."""
    src = Image.open(ICON_SRC).convert("RGB")
    w, h = src.size
    px = src.load()
    dark = lambda p: p[0] < 40 and p[1] < 40 and p[2] < 40
    x0 = next(x for x in range(w) if not dark(px[x, h // 2]))
    x1 = next(x for x in range(w - 1, -1, -1) if not dark(px[x, h // 2]))
    y0 = next(y for y in range(h) if not dark(px[w // 2, y]))
    y1 = next(y for y in range(h - 1, -1, -1) if not dark(px[w // 2, y]))
    inset = round((x1 - x0) * 0.045)
    tile = src.crop((x0 + inset, y0 + inset, x1 - inset, y1 - inset)).resize(
        (size, size), Image.LANCZOS,
    )

    # Supersampled corner mask — a 1:1 rounded rectangle has visibly stepped corners at these
    # sizes, and this artwork gets looked at large.
    ss = 4
    mask = Image.new("L", (size * ss, size * ss), 0)
    ImageDraw.Draw(mask).rounded_rectangle(
        (0, 0, size * ss - 1, size * ss - 1), radius=int(size * ss * 0.22), fill=255,
    )
    mask = mask.resize((size, size), Image.LANCZOS)
    out = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    out.paste(tile, (0, 0), mask)
    return out


def paste_icon(canvas, icon, cx, cy):
    """Icon with a soft purple halo behind it, centred on (cx, cy)."""
    s = icon.width
    pad = int(s * 0.55)
    glow = Image.new("RGBA", (s + pad * 2, s + pad * 2), (0, 0, 0, 0))
    ImageDraw.Draw(glow).rounded_rectangle(
        (pad, pad, pad + s, pad + s), radius=int(s * 0.22), fill=PURPLE + (150,),
    )
    glow = glow.filter(ImageFilter.GaussianBlur(s * 0.14))
    canvas.alpha_composite(glow, (cx - s // 2 - pad, cy - s // 2 - pad))
    canvas.alpha_composite(icon, (cx - s // 2, cy - s // 2))


def centred(draw, text, fnt, cx, y, fill):
    w = draw.textbbox((0, 0), text, font=fnt)[2]
    draw.text((cx - w // 2, y), text, font=fnt, fill=fill)


def compose(W, H, icon_frac, name_frac, gap_frac, caption=True, max_w=0.46):
    """One canvas: background, icon, wordmark, caption — stacked and centred as a block."""
    canvas = background(W, H).convert("RGBA")
    draw = ImageDraw.Draw(canvas)

    short = min(W, H)
    icon_size = int(short * icon_frac)
    name_size = int(short * name_frac)
    cap_size = max(12, int(name_size * 0.26))

    f_name = font(FONT_BLACK, name_size)
    f_cap = font(FONT_LIGHT, cap_size)

    # Shrink to fit rather than trusting the fraction. "Vista Pictures VR" is nearly twice the
    # width of a one-word name at the same point size, and on the narrow canvases a fixed
    # fraction runs it into both edges. Measured and corrected, so no canvas is a special case.
    # Measured against the QuestFiles artwork that passed Meta's review: there the content sits
    # inside the central 40% of the width. Meta's uploader draws a "safe area" box and warns
    # about anything crossing it, because these assets get cropped differently on the shelf, in
    # the app and on the web. Passed in per format — a 3000x900 banner has far less room than a
    # square, in the only direction that matters.
    MAX_W = max_w
    for _ in range(24):
        if draw.textbbox((0, 0), WORDMARK, font=f_name)[2] <= W * MAX_W:
            break
        name_size = int(name_size * 0.95)
        f_name = font(FONT_BLACK, name_size)
    for _ in range(24):
        if not caption or draw.textbbox((0, 0), CAPTION, font=f_cap)[2] <= W * MAX_W:
            break
        cap_size = int(cap_size * 0.95)
        f_cap = font(FONT_LIGHT, cap_size)

    gap = int(short * gap_frac)
    name_h = draw.textbbox((0, 0), WORDMARK, font=f_name)[3]
    cap_h = draw.textbbox((0, 0), CAPTION, font=f_cap)[3] if caption else 0
    block = icon_size + gap + name_h + (int(gap * 0.55) + cap_h if caption else 0)

    top = (H - block) // 2
    paste_icon(canvas, icon_tile(icon_size), W // 2, top + icon_size // 2)

    y = top + icon_size + gap
    centred(draw, WORDMARK, f_name, W // 2, y, TEXT)
    if caption:
        centred(draw, CAPTION, f_cap, W // 2, y + name_h + int(gap * 0.55), CYAN + (235,))
    return canvas.convert("RGB")


# name -> (W, H, icon fraction of short edge, wordmark fraction, gap fraction, caption?)
SIZES = {
    "hero_3000x900":       (3000, 900, 0.26, 0.115, 0.075, True, 0.30),
    "header_1200x630":     (1200, 630, 0.28, 0.115, 0.070, True, 0.46),
    "poster_2880x1620":    (2880, 1620, 0.24, 0.100, 0.065, True, 0.38),
    "landscape_2560x1440": (2560, 1440, 0.24, 0.100, 0.065, True, 0.38),
    "square_1440x1440":    (1440, 1440, 0.30, 0.098, 0.070, True, 0.46),
    "vertical_1008x1440":  (1008, 1440, 0.34, 0.105, 0.075, True, 0.50),
}


def main():
    os.makedirs(OUT, exist_ok=True)

    for name, (W, H, i, n, g, cap, mw) in SIZES.items():
        compose(W, H, i, n, g, cap, mw).save(os.path.join(OUT, name + ".png"), "PNG", optimize=True)
        print(f"  {name}.png  {W}x{H}")

    # App icon on its own, square and full-bleed — this one is the icon, not artwork about it.
    src = Image.open(ICON_SRC).convert("RGB")
    for px in (1024, 512, 256, 180):
        src.resize((px, px), Image.LANCZOS).save(
            os.path.join(OUT, f"icon_app_{px}.png"), "PNG", optimize=True,
        )
        print(f"  icon_app_{px}.png  {px}x{px}")

    print(f"\nWritten to {OUT}")


if __name__ == "__main__":
    main()
