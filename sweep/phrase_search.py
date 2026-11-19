"""Score new CLIP phrases against the stored sweep embeddings (no GPU, no re-render).
    python3 sweep/phrase_search.py "a photo of Elon Musk" "Pikachu" ...
For each phrase and layer: shift = mean over the four prompts of (score after push - score before), x100;
z = (shift - mean of the 128 random pushes at that dose) / (their std + 0.05). Prints the top neurons per phrase."""
import sys, json, numpy as np, warnings
warnings.filterwarnings("ignore")
import torch
from transformers import CLIPModel, CLIPTokenizer
phrases=sys.argv[1:]
m=CLIPModel.from_pretrained("openai/clip-vit-base-patch32").eval(); tok=CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
with torch.no_grad():
    t=m.get_text_features(**tok(phrases,padding=True,return_tensors="pt")); t=t/t.norm(dim=-1,keepdim=True)
T=np.array(t.tolist(),dtype=np.float32)            # [P,512]
for L in ["L10","L18","L46","L68"]:
    E=np.load(f"runs/{L}.emb.npy").astype(np.float32); E/=np.linalg.norm(E,axis=1,keepdims=True)+1e-8
    ix=np.load(f"runs/{L}.index.npz"); kind,neuron,prompt,dose=ix["kind"],ix["neuron"],ix["prompt"],ix["dose"]
    S=100*E@T.T                                      # [N,P]
    base={p:S[(kind=="baseline")&(prompt==p)][0] for p in np.unique(prompt)}
    D=S-np.stack([base[p] for p in prompt])          # change per row
    out={}
    for d in np.unique(dose[kind!="baseline"]):
        for k in ["random","neuron"]:
            sel=(kind==k)&(dose==d); u=neuron[sel]; dd=D[sel]
            uu,inv=np.unique(u,return_inverse=True); acc=np.zeros((len(uu),len(phrases))); cnt=np.zeros(len(uu))
            np.add.at(acc,inv,dd); np.add.at(cnt,inv,1); shift=acc/cnt[:,None]
            if k=="random": mu,sd=shift.mean(0),shift.std(0)
            else:
                z=(shift-mu)/(sd+0.05)
                for j,ph in enumerate(phrases):
                    for i in np.argsort(-z[:,j])[:5]: out.setdefault(ph,[]).append((float(z[i,j]),float(shift[i,j]),int(uu[i]),float(d)))
    print(f"== {L}")
    for ph in phrases:
        top=sorted(out[ph],reverse=True)[:3]
        print(f"  {ph:28s}", "  ".join(f"N{n}@{d:+g} z{z:5.1f} shift{s:+5.1f}" for z,s,n,d in top))
