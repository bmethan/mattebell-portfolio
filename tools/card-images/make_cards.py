"""Usage: python tools/card-images/make_cards.py (Pillow). Add a still's name to NAMES and its widths to CARD_WIDTHS in
components/Work.tsx.

Web-sized copies of the project card stills: WebP at 640/960/1280/1920 px wide (never above the original),
Lanczos downscale, color profile kept. Originals stay untouched in public/images."""
import os
from PIL import Image

SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'public', 'images')
OUT = f'{SRC}/cards'
NAMES = ['rpocard', 'im3card', 'realsteelcard', 'startrekcard', 'lotcard', 'walmart-famous-visitors']
WIDTHS = [640, 960, 1280, 1920]
os.makedirs(OUT, exist_ok=True)
total_in = total_out = 0
for n in NAMES:
    path = f'{SRC}/{n}.jpg'
    im = Image.open(path)
    icc = im.info.get('icc_profile')
    im = im.convert('RGB')
    total_in += os.path.getsize(path)
    made = []
    for w in WIDTHS:
        if w > im.width:
            continue
        h = round(im.height * w / im.width)
        out = f'{OUT}/{n}-{w}.webp'
        im.resize((w, h), Image.LANCZOS).save(out, 'WEBP', quality=82, method=6, **({'icc_profile': icc} if icc else {}))
        made.append(f'{w}w {os.path.getsize(out) // 1024} KB')
    print(f'{n:26} {im.width}x{im.height} {os.path.getsize(path) // 1024} KB, icc {"yes" if icc else "no"} -> ' + ', '.join(made))
