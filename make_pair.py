"""One scene, before and after, from a sweep's thumbs folder.
    python3 make_pair.py <run> <neuron> <dose> <scene> "<title>" "<line>" <out.jpg>
"""
import sys
from PIL import Image, ImageDraw, ImageFont
run, n, dose, scene, title, line, out = sys.argv[1:8]
BG=(250,249,246); INK=(28,28,28); MUTE=(110,110,110)
def font(sz,bold=False): return ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc",sz,index=1 if bold else 0)
T=512; gap=12; left=12; top=96
im=Image.new("RGB",(left+2*(T+gap),top+T+40),BG); d=ImageDraw.Draw(im)
d.text((12,16),title,fill=INK,font=font(30,True)); d.text((12,56),line,fill=MUTE,font=font(18))
im.paste(Image.open(f"thumbs/{run}/b-1_{scene}_+0.jpg").convert("RGB").resize((T,T),Image.LANCZOS),(left,top))
im.paste(Image.open(f"thumbs/{run}/n{n}_{scene}_{dose}.jpg").convert("RGB").resize((T,T),Image.LANCZOS),(left+T+gap,top))
d.text((left,top+T+8),"before",fill=MUTE,font=font(16)); d.text((left+T+gap,top+T+8),"after",fill=MUTE,font=font(16))
im.save(out,quality=92); print(out)
