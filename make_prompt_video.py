"""One prompt, many neurons: square quick-flip clip for X.   python3 make_prompt_video.py <scene> <out.mp4>
Scenes and their neuron order live in SCENES below."""
import sys, os, shutil, subprocess, tempfile
from PIL import Image, ImageDraw, ImageFont
BG=(250,249,246); INK=(28,28,28); MUTE=(110,110,110); ACC=(190,60,40)
def font(sz,bold=False): return ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc",sz,index=1 if bold else 0)
W=H=1080; S=820; X=(W-S)//2; Y=225; FPS=24
HOLD0, HOLD, FADE, END = 1.5, 0.85, 0.15, 2.5
LINK="sumansigdel.com/golden-gate-sdxl"
# name -> path template ({s} = scene)
N={"Eye neuron":"gallery/eye_full/{s}_steered.jpg","Buzz Lightyear neuron":"gallery/buzz_full/{s}_steered.jpg","Bird of prey neuron":"gallery/bird_full/{s}_steered.jpg",
 "Ghibli neuron":"gallery/anime_full/{s}_steered.jpg","Old face neuron":"gallery/oldface_full/{s}_steered.jpg","Horror neuron":"gallery/horror_full/{s}_steered.jpg",
 "Dog neuron":"gallery/dog/{s}_steered.jpg","Drone shot neuron":"gallery/n976p/{s}_steered.jpg","Golden Gate (two neurons)":"gallery/gg3_full/{s}_steered.jpg",
 "Skyscraper neuron":"thumbs/demo-new3-L46/n1245_{s}_+2.jpg","Basketball neuron":"thumbs/demo-new-L46/n4630_{s}_-2.jpg","Gym bro neuron":"thumbs/demo-weird-L18/n2298_{s}_-2.jpg",
 "Lips neuron":"thumbs/demo-weird-L18/n4168_{s}_+2.jpg","Kusama neuron":"thumbs/demo-new3-L46/n2240_{s}_+2.jpg","Zebra pattern neuron":"thumbs/demo-new3-L46/n1620_{s}_-2.jpg",
 "White flowers neuron":"thumbs/demo-new-L46/n456_{s}_-2.jpg","Van Gogh neuron":"thumbs/demo-new3-L46/n2063_{s}_+2.jpg","Herd neuron":"thumbs/demo-new3-L18/n3204_{s}_+2.jpg",
 "Faces everywhere neuron":"thumbs/demo-new3-L46/n4429_{s}_-2.jpg","Crowd of faces neuron":"thumbs/demo-weird-L46/n5_{s}_+1.jpg","Car neuron":"thumbs/demo-new-L18/n1232_{s}_+2.jpg",
 "Woman neuron":"thumbs/demo-new3-L18/n4782_{s}_+2.jpg","Fur neuron":"thumbs/demo-new-L46/n3969_{s}_-2.jpg","Barrels neuron":"thumbs/demo-weird-L18/n568_{s}_-2.jpg","Persian rug neuron":"thumbs/demo-weird-L18/n2886_{s}_-2.jpg",
 "Clock face neuron":"thumbs/demo-new-L18/n3569_{s}_+2.jpg","Ink drawing neuron":"thumbs/demo-new3-L46/n1565_{s}_-2.jpg","Mona Lisa neuron":"thumbs/famous-L18/n216_{s}_+2.jpg","Old master neuron":"thumbs/famous-L18/n3641_{s}_-2.jpg","New York neuron":"thumbs/famous-L18/n938_{s}_-2.jpg","Christmas neuron":"thumbs/famous-L46/n2894_{s}_+1.jpg","Fairy lights neuron":"thumbs/famous-L46/n2174_{s}_-1.jpg"}

FD="/System/Library/Fonts/Supplemental/"
STYLE={  # name -> (font file, index, colour)
 "Eye neuron":("Didot.ttc",2,(60,60,60)),"Buzz Lightyear neuron":("Phosphate.ttc",0,(70,150,60)),"Bird of prey neuron":("Papyrus.ttc",0,(120,80,30)),
 "Ghibli neuron":("Comic Sans MS Bold.ttf",0,(220,60,140)),"Old face neuron":("Baskerville.ttc",0,(90,90,90)),"Horror neuron":("Chalkduster.ttf",0,(150,20,20)),
 "Dog neuron":("Chalkboard.ttc",0,(200,120,30)),"Drone shot neuron":("Courier New Bold.ttf",0,(30,110,120)),"Golden Gate (two neurons)":("Copperplate.ttc",1,(200,70,30)),
 "Skyscraper neuron":("Futura.ttc",2,(70,90,120)),"Basketball neuron":("Arial Black.ttf",0,(230,110,30)),"Gym bro neuron":("Impact.ttf",0,(190,40,40)),
 "Lips neuron":("Arial Black.ttf",0,(200,30,70)),"Kusama neuron":("Bradley Hand Bold.ttf",0,(40,40,40)),"Zebra pattern neuron":("Impact.ttf",0,(20,20,20)),
 "White flowers neuron":("Zapfino.ttf",0,(120,130,90)),"Van Gogh neuron":("Brush Script.ttf",0,(40,90,200)),"Herd neuron":("Rockwell.ttc",1,(170,40,40)),
 "Faces everywhere neuron":("Trattatello.ttf",0,(110,70,60)),"Crowd of faces neuron":("Trattatello.ttf",0,(110,70,60)),"Car neuron":("Futura.ttc",1,(200,40,40)),
 "Woman neuron":("Didot.ttc",2,(160,60,120)),"Fur neuron":("Chalkduster.ttf",0,(120,80,50)),"Barrels neuron":("Rockwell.ttc",1,(120,70,30)),"Persian rug neuron":("Luminari.ttf",0,(150,40,40)),
 "Clock face neuron":("Copperplate.ttc",1,(70,70,70)),"Ink drawing neuron":("SignPainter.ttc",0,(40,40,40)),"Mona Lisa neuron":("Didot.ttc",2,(110,80,40)),"Old master neuron":("Baskerville.ttc",2,(90,70,50)),"New York neuron":("Futura.ttc",2,(40,40,40)),"Christmas neuron":("Chalkboard.ttc",1,(180,30,30)),"Fairy lights neuron":("Luminari.ttf",0,(200,150,40))}
def title_font(name, size):
    f,i,_=STYLE.get(name,("",0,ACC))
    try: return ImageFont.truetype(FD+f,size,index=i)
    except Exception: return font(size,True)

SCENES={
 "cat":("a cat sleeping on a windowsill",["Eye neuron","Buzz Lightyear neuron","Zebra pattern neuron","Persian rug neuron","Basketball neuron","Bird of prey neuron","Van Gogh neuron","Kusama neuron","Skyscraper neuron","Ghibli neuron","Drone shot neuron"]),
 "street":("a Tokyo street at night",["Buzz Lightyear neuron","Herd neuron","Zebra pattern neuron","Golden Gate (two neurons)","Ghibli neuron","Basketball neuron","Persian rug neuron","Kusama neuron","Van Gogh neuron","Bird of prey neuron","Drone shot neuron"]),
 "breakfast":("pancakes with syrup and berries",["Eye neuron","Basketball neuron","Ghibli neuron","Car neuron","Zebra pattern neuron","Persian rug neuron","Buzz Lightyear neuron","Kusama neuron","Van Gogh neuron","Skyscraper neuron"]),
 "portrait":("an old man with a grey beard",["Eye neuron","Buzz Lightyear neuron","Zebra pattern neuron","Basketball neuron","Bird of prey neuron","Kusama neuron","Van Gogh neuron","Skyscraper neuron","White flowers neuron","Dog neuron"]),
 "elon":("a photo of Elon Musk",["Buzz Lightyear neuron","Mona Lisa neuron","Zebra pattern neuron","Ghibli neuron","Christmas neuron","Kusama neuron","Bird of prey neuron","Basketball neuron","Car neuron","Van Gogh neuron","Dog neuron","Skyscraper neuron","Herd neuron","Ink drawing neuron"]),
 "zuck":("a photo of Mark Zuckerberg",["Eye neuron","Buzz Lightyear neuron","Mona Lisa neuron","Zebra pattern neuron","Basketball neuron","Ghibli neuron","Kusama neuron","Van Gogh neuron","Skyscraper neuron","Christmas neuron","Old master neuron","Bird of prey neuron","Drone shot neuron"]),
 "dog":("a dog running on a beach",["Buzz Lightyear neuron","Ghibli neuron","Bird of prey neuron","Basketball neuron","Zebra pattern neuron","Kusama neuron","Van Gogh neuron","Skyscraper neuron","Golden Gate (two neurons)"]),
}
FAMOUS={"elon","zuck"}
def path_for(name, scene):
    t=N[name]
    if scene in FAMOUS:
        if t.startswith("gallery/"):           # old curated galleries: map by neuron id
            G={"Eye neuron":("famous-L18",297,"-2"),"Buzz Lightyear neuron":("famous-L18",1505,"-2"),"Bird of prey neuron":("famous-L18",1532,"-2"),"Ghibli neuron":("famous-L18",300,"-2"),
               "Old face neuron":("famous-L18",3462,"-2"),"Horror neuron":("famous-L18",5097,"-2"),"Dog neuron":("famous-L18",2683,"-2"),"Drone shot neuron":("famous-L18",976,"+2")}
            r,n,d=G[name]; return f"thumbs/{r}/n{n}_{scene}_{d}.jpg"
        return t.replace("demo-weird-L18","famous-L18").replace("demo-new-L18","famous-L18").replace("demo-new3-L18","famous-L18").replace("demo-weird-L46","famous-L46").replace("demo-new-L46","famous-L46").replace("demo-new3-L46","famous-L46").format(s=scene)
    return t.format(s=scene)
scene, out = sys.argv[1], sys.argv[2]
PROMPT, order = SCENES[scene]
def frame(img, title, dim=False):
    fr=Image.new("RGB",(W,H),BG); d=ImageDraw.Draw(fr)
    f=font(34,True); t="\u201c"+PROMPT+"\u201d"; d.text(((W-d.textlength(t,font=f))//2,56),t,fill=INK,font=f)
    f=font(30,True); d.text(((W-d.textlength("+",font=f))//2,104),"+",fill=INK,font=f)
    col=MUTE if dim else STYLE.get(title,("",0,ACC))[2]; size=(40 if dim else 58)
    f=title_font(title,size) if not dim else font(40,True)
    while d.textlength(title,font=f)>W-120 and size>30: size-=4; f=title_font(title,size)
    bb=d.textbbox((0,0),title,font=f); tw=bb[2]-bb[0]; th=bb[3]-bb[1]
    d.text(((W-tw)//2-bb[0],138+(60-th)//2-bb[1]),title,fill=col,font=f)
    fr.paste(img,(X,Y))
    return fr
def load(p): return Image.open(p).convert("RGB").resize((S,S),Image.LANCZOS)
tmp=tempfile.mkdtemp(); n=0
def put(fr):
    global n; fr.save(f"{tmp}/f{n:05d}.jpg",quality=92); n+=1
prev=load(f"thumbs/famous-L18/b-1_{scene}_+0.jpg" if scene in FAMOUS else f"gallery/eye_full/{scene}_base.jpg"); last_title="no neuron pushed"
for _ in range(int(HOLD0*FPS)): put(frame(prev,last_title,dim=True))
for title in order:
    cur=load(path_for(title,scene)); nf=int(FADE*FPS)
    for i in range(nf): put(frame(Image.blend(prev,cur,(i+1)/nf),title))
    for _ in range(int(HOLD*FPS)): put(frame(cur,title))
    prev, last_title = cur, title
end=Image.new("RGB",(W,H),BG); d=ImageDraw.Draw(end); f=font(34,True); d.text(((W-d.textlength(LINK,font=f))//2,H//2-20),LINK,fill=INK,font=f)
nf=int(FADE*FPS)
for i in range(nf): put(Image.blend(frame(prev,last_title),end,(i+1)/nf))
for _ in range(int(END*FPS)): put(end)
subprocess.run(["ffmpeg","-y","-loglevel","error","-framerate",str(FPS),"-i",f"{tmp}/f%05d.jpg","-c:v","libx264","-pix_fmt","yuv420p","-crf","20","-movflags","+faststart",out],check=True)
shutil.rmtree(tmp); print(out, f"{n/FPS:.1f}s", os.path.getsize(out)//1024, "KB")
