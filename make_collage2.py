"""Twelve-scene before/after collage straight from a sweep's thumbs folder.
    python3 make_collage2.py <run> <neuron> <dose> "<title>" "<line>" <out.jpg>
"""
import sys, os
from PIL import Image, ImageDraw, ImageFont
run, n, dose, title, line, out = sys.argv[1:7]
BG=(250,249,246); INK=(28,28,28); MUTE=(110,110,110)
def font(sz,bold=False): return ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc",sz,index=1 if bold else 0)
scenes=["astronaut","boat","breakfast","castle","cat","desert","dog","forest","kitchen","library","portrait","street"]
T=300; gap=12; left=12; top=96; cols=6
W=left+cols*(T+gap); H=top+2*(2*T+gap+40)+10
im=Image.new("RGB",(W,H),BG); d=ImageDraw.Draw(im)
d.text((12,16),title,fill=INK,font=font(30,True))
d.text((12,56),line+" · before above, after below",fill=MUTE,font=font(18))
for i,s in enumerate(scenes):
    r,c=divmod(i,cols); x=left+c*(T+gap); y=top+r*(2*T+gap+40)
    im.paste(Image.open(f"thumbs/{run}/b-1_{s}_+0.jpg").convert("RGB").resize((T,T),Image.LANCZOS),(x,y))
    im.paste(Image.open(f"thumbs/{run}/n{n}_{s}_{dose}.jpg").convert("RGB").resize((T,T),Image.LANCZOS),(x,y+T+gap))
    d.text((x,y+2*T+gap+6),s,fill=MUTE,font=font(15))
im.save(out,quality=90); print(out, im.size)
