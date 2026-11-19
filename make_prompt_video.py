"""One prompt, many neurons: square quick-flip clip for X.   python3 make_prompt_video.py <scene> <out.mp4>
Scenes and their neuron order live in SCENES below."""
import sys, os, shutil, subprocess, tempfile
from PIL import Image, ImageDraw, ImageFont
BG=(250,249,246); INK=(28,28,28); MUTE=(110,110,110); ACC=(190,60,40)
def font(sz,bold=False): return ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc",sz,index=1 if bold else 0)
W=H=1080; S=820; X=(W-S)//2; Y=225; FPS=24
HOLD0, HOLD, FADE, END = 1.5, 0.8, 0.15, 2.5
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
 "Clock face neuron":"thumbs/demo-new-L18/n3569_{s}_+2.jpg","Ink drawing neuron":"thumbs/demo-new3-L46/n1565_{s}_-2.jpg","Mona Lisa neuron":"thumbs/famous-L18/n216_{s}_+2.jpg","Old master neuron":"thumbs/famous-L18/n3641_{s}_-2.jpg","New York neuron":"thumbs/famous-L18/n938_{s}_-2.jpg","Christmas neuron":"thumbs/famous-L46/n2894_{s}_+1.jpg","Fairy lights neuron":"thumbs/famous-L46/n2174_{s}_-1.jpg",
 "Plush toy neuron":"thumbs/famous2-L18/n3771_{s}_-2.jpg","Bronze medallion neuron":"thumbs/famous2-L18/n4249_{s}_-2.jpg","Oscar statue neuron":"thumbs/famous2-L18/n343_{s}_+2.jpg",
 "Tattoo neuron":"thumbs/famous2-L18/n1567_{s}_-2.jpg","Mannequin neuron":"thumbs/famous2-L18/n3001_{s}_-2.jpg","Cowboy hat neuron":"thumbs/famous2-L18/n4034_{s}_-2.jpg",
 "Gorilla neuron":"thumbs/famous2-L18/n4868_{s}_+2.jpg","Mech robot neuron":"thumbs/famous2-L18/n2987_{s}_+2.jpg","Funko Pop neuron":"thumbs/famous2-L18/n3335_{s}_+2.jpg",
 "3D cartoon neuron":"thumbs/famous2-L18/n1226_{s}_-2.jpg","Comic strip neuron":"thumbs/famous2-L18/n3704_{s}_+2.jpg","Cake neuron":"thumbs/famous2-L18/n4096_{s}_-2.jpg",
 "Beard neuron":"thumbs/famous2-L18/n2385_{s}_+2.jpg","Gilded frame neuron":"thumbs/famous2-L18/n893_{s}_+2.jpg","Pop art neuron":"thumbs/famous2-L46/n2384_{s}_+2.jpg",
 "Lego neuron":"thumbs/famous2-L46/n888_{s}_+2.jpg","Autumn neuron":"thumbs/famous2-L46/n3410_{s}_+1.jpg","Wrought iron neuron":"thumbs/famous2-L46/n4422_{s}_+2.jpg",
 "American flag neuron":"thumbs/famous2-L46/n3631_{s}_-2.jpg","Sand neuron":"thumbs/famous2-L46/n1691_{s}_+2.jpg","Marble relief neuron":"thumbs/famous2-L46/n417_{s}_-2.jpg",
 "Wood engraving neuron":"thumbs/famous2-L46/n5017_{s}_-2.jpg","Knitted neuron":"thumbs/famous2-L10/n1419_{s}_-2.jpg","QR code neuron":"thumbs/famous2-L10/n3066_{s}_+2.jpg","Bronze statue neuron":"thumbs/famous2-L46/n3472_{s}_+2.jpg","Sketch neuron":"thumbs/famous2-L46/n4158_{s}_+2.jpg",
 "Magenta neuron":"thumbs/famous2-L68/n953_{s}_+1.jpg","Purple neuron":"thumbs/famous2-L68/n66_{s}_+1.jpg",
 "Banknote neuron":"thumbs/famous3-L18/n1342_{s}_-2.jpg","Plaque neuron":"thumbs/famous3-L18/n1172_{s}_+2.jpg","Stadium neuron":"thumbs/famous3-L18/n767_{s}_+2.jpg",
 "Car interior neuron":"thumbs/famous3-L18/n1625_{s}_+2.jpg","Horse neuron":"thumbs/famous3-L18/n2333_{s}_-2.jpg","Arena neuron":"thumbs/famous3-L18/n4165_{s}_-2.jpg",
 "Colonnade neuron":"thumbs/famous3-L18/n2486_{s}_-2.jpg","Cathedral neuron":"thumbs/famous3-L18/n4126_{s}_+2.jpg","Map neuron":"thumbs/famous3-L18/n4584_{s}_+2.jpg",
 "Roar neuron":"thumbs/famous3-L18/n1529_{s}_-2.jpg","Airplane neuron":"thumbs/famous3-L18/n1480_{s}_-2.jpg","Camera neuron":"thumbs/famous3-L18/n1711_{s}_-2.jpg",
 "Blossom neuron":"thumbs/famous3-L18/n935_{s}_-2.jpg","Gift box neuron":"thumbs/famous3-L18/n148_{s}_+2.jpg","Envelope neuron":"thumbs/famous3-L18/n4403_{s}_-2.jpg",
 "Gym neuron":"thumbs/famous3-L18/n4836_{s}_+2.jpg","Lion neuron":"thumbs/famous3-L18/n3392_{s}_-2.jpg","Dinner plate neuron":"thumbs/famous3-L18/n4874_{s}_-2.jpg",
 "Hillside village neuron":"thumbs/famous3-L18/n3048_{s}_-2.jpg","Wedding dress neuron":"thumbs/famous3-L18/n4148_{s}_+2.jpg","Scaffolding neuron":"thumbs/famous3-L18/n707_{s}_+2.jpg",
 "Lavender neuron":"thumbs/famous3-L46/n4564_{s}_+2.jpg","Moss neuron":"thumbs/famous3-L46/n249_{s}_+2.jpg","Stone carving neuron":"thumbs/famous3-L46/n1921_{s}_-2.jpg","Pillow neuron":"thumbs/famous3-L18/n4403_{s}_-2.jpg","Courtroom neuron":"thumbs/famous3-L18/n4165_{s}_-2.jpg",
 "Sticky notes neuron":"thumbs/famous3-L46/n1569_{s}_-2.jpg","Tile neuron":"thumbs/famous3-L46/n3506_{s}_+2.jpg","Union Jack neuron":"thumbs/famous3-L46/n4628_{s}_+2.jpg",
 "Lemon neuron":"thumbs/famous3-L46/n4071_{s}_-1.jpg","Ember neuron":"thumbs/famous3-L68/n524_{s}_+2.jpg",
 "Spreadsheet neuron":"thumbs/famous3b-L18/n2356_{s}_-2.jpg","Website neuron":"thumbs/famous3b-L18/n225_{s}_-2.jpg","Magazine cover neuron":"thumbs/famous3b-L18/n1622_{s}_+2.jpg",
 "Collage neuron":"thumbs/famous3b-L18/n625_{s}_-2.jpg","Contact sheet neuron":"thumbs/famous3b-L18/n2934_{s}_-2.jpg","Diagram neuron":"thumbs/famous3b-L18/n3254_{s}_-2.jpg",
 "Movie poster neuron":"thumbs/famous3b-L18/n4597_{s}_+2.jpg","Old photo neuron":"thumbs/famous3b-L18/n801_{s}_+2.jpg","Pencil sketch neuron":"thumbs/famous3b-L18/n3719_{s}_-1.jpg",
 "Passport photo neuron":"thumbs/famous3b-L18/n3767_{s}_+2.jpg","Tiny Elon neuron":"thumbs/famous3b-L18/n3487_{s}_-2.jpg","Shop window neuron":"thumbs/famous3b-L18/n3743_{s}_+2.jpg",
 "Comic page neuron":"thumbs/famous3b-L46/n1383_{s}_-2.jpg","Video game neuron":"thumbs/famous3b-L46/n4899_{s}_-2.jpg"}

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
 "Clock face neuron":("Copperplate.ttc",1,(70,70,70)),"Ink drawing neuron":("SignPainter.ttc",0,(40,40,40)),"Mona Lisa neuron":("Didot.ttc",2,(110,80,40)),"Old master neuron":("Baskerville.ttc",2,(90,70,50)),"New York neuron":("Futura.ttc",2,(40,40,40)),"Christmas neuron":("Chalkboard.ttc",1,(180,30,30)),"Fairy lights neuron":("Luminari.ttf",0,(200,150,40)),
 "Plush toy neuron":("Chalkboard.ttc",1,(220,80,60)),"Bronze medallion neuron":("Copperplate.ttc",1,(150,100,40)),"Oscar statue neuron":("Didot.ttc",2,(190,150,40)),
 "Tattoo neuron":("Impact.ttf",0,(40,40,40)),"Mannequin neuron":("Futura.ttc",2,(120,120,120)),"Cowboy hat neuron":("Rockwell.ttc",1,(140,80,30)),
 "Gorilla neuron":("Impact.ttf",0,(60,40,30)),"Mech robot neuron":("Futura.ttc",2,(200,40,40)),"Funko Pop neuron":("Arial Black.ttf",0,(230,120,30)),
 "3D cartoon neuron":("Comic Sans MS Bold.ttf",0,(40,120,200)),"Comic strip neuron":("Comic Sans MS Bold.ttf",0,(200,40,40)),"Cake neuron":("Chalkboard.ttc",1,(200,60,120)),
 "Beard neuron":("Rockwell.ttc",1,(90,60,30)),"Gilded frame neuron":("Didot.ttc",2,(160,120,30)),"Pop art neuron":("Arial Black.ttf",0,(220,50,120)),
 "Lego neuron":("Arial Black.ttf",0,(220,40,40)),"Autumn neuron":("Baskerville.ttc",2,(200,100,30)),"Wrought iron neuron":("Trattatello.ttf",0,(40,40,40)),
 "American flag neuron":("Impact.ttf",0,(180,30,50)),"Sand neuron":("Papyrus.ttc",0,(170,130,80)),"Marble relief neuron":("Didot.ttc",0,(110,110,110)),
 "Wood engraving neuron":("Trattatello.ttf",0,(90,60,40)),"Knitted neuron":("Chalkduster.ttf",0,(180,60,80)),"QR code neuron":("Courier New Bold.ttf",0,(30,30,30)),"Bronze statue neuron":("Copperplate.ttc",1,(140,90,40)),"Sketch neuron":("Bradley Hand Bold.ttf",0,(60,60,60)),
 "Magenta neuron":("Phosphate.ttc",0,(220,20,120)),"Purple neuron":("Phosphate.ttc",0,(130,50,180)),
 "Banknote neuron":("Copperplate.ttc",1,(40,90,60)),"Plaque neuron":("Copperplate.ttc",1,(120,60,40)),"Stadium neuron":("Impact.ttf",0,(40,80,160)),
 "Car interior neuron":("Futura.ttc",2,(160,40,40)),"Horse neuron":("Rockwell.ttc",1,(120,70,40)),"Arena neuron":("Rockwell.ttc",1,(150,60,40)),
 "Colonnade neuron":("Didot.ttc",2,(90,90,90)),"Cathedral neuron":("Luminari.ttf",0,(120,60,120)),"Map neuron":("Courier New Bold.ttf",0,(200,40,40)),
 "Roar neuron":("Impact.ttf",0,(180,30,30)),"Airplane neuron":("Futura.ttc",2,(40,100,170)),"Camera neuron":("Arial Black.ttf",0,(40,40,40)),
 "Blossom neuron":("Zapfino.ttf",0,(200,100,140)),"Gift box neuron":("Chalkboard.ttc",1,(200,40,60)),"Envelope neuron":("American Typewriter.ttc",1,(80,80,80)),
 "Gym neuron":("Impact.ttf",0,(60,60,60)),"Lion neuron":("Impact.ttf",0,(170,110,30)),"Dinner plate neuron":("Didot.ttc",2,(60,60,60)),
 "Hillside village neuron":("Baskerville.ttc",2,(80,110,60)),"Wedding dress neuron":("SnellRoundhand.ttc",1,(170,120,140)),"Scaffolding neuron":("Futura.ttc",2,(90,90,90)),
 "Lavender neuron":("Zapfino.ttf",0,(130,80,180)),"Moss neuron":("Chalkduster.ttf",0,(70,120,50)),"Stone carving neuron":("Trattatello.ttf",0,(90,90,90)),"Pillow neuron":("Chalkboard.ttc",1,(150,120,100)),"Courtroom neuron":("Copperplate.ttc",1,(80,60,40)),
 "Sticky notes neuron":("Bradley Hand Bold.ttf",0,(200,160,30)),"Tile neuron":("Futura.ttc",2,(60,60,60)),"Union Jack neuron":("Impact.ttf",0,(30,50,140)),
 "Lemon neuron":("Chalkboard.ttc",1,(200,170,30)),"Ember neuron":("Chalkduster.ttf",0,(200,70,20)),
 "Spreadsheet neuron":("Courier New Bold.ttf",0,(30,120,60)),"Website neuron":("Courier New Bold.ttf",0,(30,90,200)),"Magazine cover neuron":("Didot.ttc",2,(200,30,60)),
 "Collage neuron":("Bradley Hand Bold.ttf",0,(60,60,60)),"Contact sheet neuron":("Courier New Bold.ttf",0,(60,60,60)),"Diagram neuron":("Courier New Bold.ttf",0,(60,60,60)),
 "Movie poster neuron":("Impact.ttf",0,(200,160,30)),"Old photo neuron":("American Typewriter.ttc",1,(110,90,70)),"Pencil sketch neuron":("Bradley Hand Bold.ttf",0,(70,70,70)),
 "Passport photo neuron":("American Typewriter.ttc",1,(60,60,60)),"Tiny Elon neuron":("Comic Sans MS Bold.ttf",0,(60,60,60)),"Shop window neuron":("Futura.ttc",2,(90,90,90)),
 "Comic page neuron":("Comic Sans MS Bold.ttf",0,(200,40,40)),"Video game neuron":("Courier New Bold.ttf",0,(40,160,80))}
def title_font(name, size):
    f,i,_=STYLE.get(name,("",0,ACC))
    try: return ImageFont.truetype(FD+f,size,index=i)
    except Exception: return font(size,True)

SCENES={
 "cat":("a cat sleeping on a windowsill",["Eye neuron","Buzz Lightyear neuron","Zebra pattern neuron","Persian rug neuron","Basketball neuron","Bird of prey neuron","Van Gogh neuron","Kusama neuron","Skyscraper neuron","Ghibli neuron","Drone shot neuron"]),
 "street":("a Tokyo street at night",["Buzz Lightyear neuron","Herd neuron","Zebra pattern neuron","Golden Gate (two neurons)","Ghibli neuron","Basketball neuron","Persian rug neuron","Kusama neuron","Van Gogh neuron","Bird of prey neuron","Drone shot neuron"]),
 "breakfast":("pancakes with syrup and berries",["Eye neuron","Basketball neuron","Ghibli neuron","Car neuron","Zebra pattern neuron","Persian rug neuron","Buzz Lightyear neuron","Kusama neuron","Van Gogh neuron","Skyscraper neuron"]),
 "portrait":("an old man with a grey beard",["Eye neuron","Buzz Lightyear neuron","Zebra pattern neuron","Basketball neuron","Bird of prey neuron","Kusama neuron","Van Gogh neuron","Skyscraper neuron","White flowers neuron","Dog neuron"]),
 "elon":("a photo of Elon Musk",['Dog neuron', 'Buzz Lightyear neuron', 'Banknote neuron', 'Funko Pop neuron', 'Magazine cover neuron', 'Mona Lisa neuron', 'Roar neuron', 'American flag neuron', 'Spreadsheet neuron', 'Zebra pattern neuron', 'Plush toy neuron', 'Diagram neuron', 'Ghibli neuron', 'Oscar statue neuron', 'Gift box neuron', 'Gym bro neuron', 'Movie poster neuron', 'Christmas neuron', 'Mech robot neuron', 'Plaque neuron', 'Kusama neuron', 'Pop art neuron', 'Tiny Elon neuron', 'Lips neuron', 'Union Jack neuron', '3D cartoon neuron', 'Contact sheet neuron', 'Bird of prey neuron', 'Marble relief neuron', 'Website neuron', 'Basketball neuron', 'QR code neuron', 'Car interior neuron', 'Car neuron', 'Bronze statue neuron', 'Old photo neuron', 'Van Gogh neuron', 'Wrought iron neuron', 'Lavender neuron', 'Old master neuron', 'Sand neuron', 'Pencil sketch neuron', 'Skyscraper neuron', 'Comic strip neuron', 'Pillow neuron', 'Cake neuron', 'Lego neuron', 'Moss neuron', 'Gilded frame neuron', 'Wood engraving neuron', 'Stadium neuron', 'Courtroom neuron', 'Ember neuron', 'Magenta neuron', 'Ink drawing neuron']),
 "zuck":("a photo of Mark Zuckerberg",["Dog neuron","Buzz Lightyear neuron","Mona Lisa neuron","Zebra pattern neuron","Ghibli neuron","Christmas neuron","Kusama neuron","Bird of prey neuron","Basketball neuron","Car neuron","Van Gogh neuron","Old master neuron","Crowd of faces neuron","Skyscraper neuron","Ink drawing neuron"]),
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
        return t.replace("demo-weird-L18","famous-L18").replace("demo-new-L18","famous-L18").replace("demo-new3-L18","famous-L18").replace("demo-weird-L46","famous-L46").replace("demo-new-L46","famous-L46").replace("demo-new3-L46","famous-L46").format(s=scene) if not t.startswith("thumbs/famous2") and not t.startswith("thumbs/famous3") and not t.startswith("thumbs/famous3b") else t.format(s=scene)
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
