"""Twitter clip: each neuron fades from off to on, with name, dose and prompt. Edit SEGS, then:
    python3 make_video.py gallery/figures/neurons.mp4
"""
import sys, os, shutil, subprocess, tempfile
from PIL import Image, ImageDraw, ImageFont
BG=(250,249,246); INK=(28,28,28); MUTE=(110,110,110); RED=(190,60,40)
def font(sz,bold=False): return ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc",sz,index=1 if bold else 0)
W,H=1280,720; S=600; X=60; Y=60; FPS=24
SEGS=[
 ("Elizabeth Holmes neuron","layer 18 · neuron 297 · pushed to −2","a medieval castle on a hill in the fog","gallery/eye_full/castle_base.jpg","gallery/eye_full/castle_steered.jpg"),
 ("Aerial view neuron","layer 18 · neuron 976 · pushed to +2","a busy Tokyo street at night with neon signs","gallery/n976p/street_base.jpg","gallery/n976p/street_steered.jpg"),
 ("Herd neuron","layer 18 · neuron 3204 · pushed to +2","a red sports car on a city street","thumbs/new3-L18/b-1_car_+0.jpg","thumbs/new3-L18/n3204_car_+2.jpg"),
 ("Basketball neuron","layer 46 · neuron 4630 · pushed to −2","a photo of a cat sleeping on a windowsill","thumbs/demo-new-L46/b-1_cat_+0.jpg","thumbs/demo-new-L46/n4630_cat_-2.jpg"),
 ("Crowd of faces neuron","layer 46 · neuron 5 · pushed to +1","a plate of pancakes with maple syrup and berries","thumbs/demo-weird-L46/b-1_breakfast_+0.jpg","thumbs/demo-weird-L46/n5_breakfast_+1.jpg"),
 ("Skyscraper neuron","layer 46 · neuron 1245 · pushed to +2","a photo of a cat sleeping on a windowsill","thumbs/demo-new3-L46/b-1_cat_+0.jpg","thumbs/demo-new3-L46/n1245_cat_+2.jpg"),
 ("Lips neuron","layer 18 · neuron 4168 · pushed to +2","a golden retriever running on a beach","thumbs/demo-weird-L18/b-1_dog_+0.jpg","thumbs/demo-weird-L18/n4168_dog_+2.jpg"),
 ("Abs neuron","layer 18 · neuron 2298 · pushed to −2","a photo of a cat sleeping on a windowsill","thumbs/demo-weird-L18/b-1_cat_+0.jpg","thumbs/demo-weird-L18/n2298_cat_-2.jpg"),
 ("Polka dot neuron","layer 46 · neuron 2240 · pushed to +2","a studio portrait of an old man with a grey beard","thumbs/demo-new3-L46/b-1_portrait_+0.jpg","thumbs/demo-new3-L46/n2240_portrait_+2.jpg"),
 ("Clock face neuron","layer 18 · neuron 3569 · pushed to +2","a wooden rowing boat on a calm river","thumbs/demo-new-L18/b-1_boat_+0.jpg","thumbs/demo-new-L18/n3569_boat_+2.jpg"),
]
HOLD_BEFORE, FADE, HOLD_AFTER = 1.0, 0.7, 1.7
def wrap(d, text, f, width):
    lines=[]; cur=""
    for w in text.split():
        t=(cur+" "+w).strip()
        if d.textlength(t,font=f)>width: lines.append(cur); cur=w
        else: cur=t
    return lines+[cur]
def panel(title, sub, prompt, img, t):
    fr=Image.new("RGB",(W,H),BG); fr.paste(img,(X,Y)); d=ImageDraw.Draw(fr); tx=X+S+56
    d.text((tx,Y+10),"Pushing one neuron",fill=MUTE,font=font(22)); d.text((tx,Y+44),"in SDXL-Turbo",fill=MUTE,font=font(22))
    yy=Y+130
    for ln in wrap(d,title,font(44,True),W-tx-40): d.text((tx,yy),ln,fill=INK,font=font(44,True)); yy+=52
    d.text((tx,yy+10),sub,fill=MUTE,font=font(20)); d.text((tx,yy+60),"prompt",fill=MUTE,font=font(16)); py=yy+84
    for ln in wrap(d,prompt,font(20),W-tx-40): d.text((tx,py),ln,fill=INK,font=font(20)); py+=26
    by=Y+S-24
    d.text((tx,by-30),"neuron off" if t<=0 else ("neuron on" if t>=1 else "pushing…"),fill=RED if t>0 else MUTE,font=font(18,True))
    d.rounded_rectangle((tx,by,tx+360,by+10),radius=5,fill=(225,222,215))
    if t>0: d.rounded_rectangle((tx,by,tx+int(360*t),by+10),radius=5,fill=RED)
    return fr
out=sys.argv[1]; tmp=tempfile.mkdtemp(); n=0
for title,sub,prompt,a,b in SEGS:
    A=Image.open(a).convert("RGB").resize((S,S),Image.LANCZOS); B=Image.open(b).convert("RGB").resize((S,S),Image.LANCZOS)
    for _ in range(int(HOLD_BEFORE*FPS)): panel(title,sub,prompt,A,0).save(f"{tmp}/f{n:05d}.jpg",quality=90); n+=1
    nf=int(FADE*FPS)
    for i in range(nf): t=(i+1)/nf; panel(title,sub,prompt,Image.blend(A,B,t),t).save(f"{tmp}/f{n:05d}.jpg",quality=90); n+=1
    for _ in range(int(HOLD_AFTER*FPS)): panel(title,sub,prompt,B,1).save(f"{tmp}/f{n:05d}.jpg",quality=90); n+=1
subprocess.run(["ffmpeg","-y","-loglevel","error","-framerate",str(FPS),"-i",f"{tmp}/f%05d.jpg","-c:v","libx264","-pix_fmt","yuv420p","-crf","20","-movflags","+faststart",out],check=True)
shutil.rmtree(tmp); print(out, f"{n/FPS:.1f}s", os.path.getsize(out)//1024, "KB")
