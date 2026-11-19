"""Twelve-scene before/after collage for one gallery: python3 make_collage.py <gallery> "<title>" "<line>" <out>"""
import sys
from PIL import Image, ImageDraw, ImageFont
g_, title, line, out = sys.argv[1:5]
f = lambda sz, b=False: ImageFont.truetype('/System/Library/Fonts/Helvetica.ttc', sz, index=1 if b else 0)
scenes = ["astronaut","boat","breakfast","castle","cat","desert","dog","forest","kitchen","library","portrait","street"]
T, g, cols, rows = 300, 12, 6, 2
W = g + cols * (T + g); H = 104 + rows * (2 * T + 6 + 34 + g)
im = Image.new('RGB', (W, H), (250, 249, 246)); d = ImageDraw.Draw(im)
d.text((g, 20), title, font=f(30, True), fill=(26, 26, 26))
d.text((g, 58), line + " · before above, after below", font=f(20), fill=(90, 90, 90))
for i, s in enumerate(scenes):
    r, c = divmod(i, cols); x = g + c * (T + g); y = 104 + r * (2 * T + 6 + 34 + g)
    im.paste(Image.open(f'gallery/{g_}/{s}_base.jpg').convert('RGB').resize((T, T), Image.LANCZOS), (x, y))
    im.paste(Image.open(f'gallery/{g_}/{s}_steered.jpg').convert('RGB').resize((T, T), Image.LANCZOS), (x, y + T + 6))
    d.text((x, y + 2 * T + 12), s, font=f(17), fill=(90, 90, 90))
im.save(out, quality=90); print(out, im.size)
