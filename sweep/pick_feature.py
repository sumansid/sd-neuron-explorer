"""Rank-1 feature id of a find_L<l>_<tag>.json, preferring the largest mean activation on the tag among the top 5
(the family head writes most strongly): python sweep/pick_feature.py sae/flux3/find_L17_gg.json"""
import json, sys
d = json.load(open(sys.argv[1])); top = d["top"][:5]
print(max(top, key=lambda f: f["mean_tag"])["feature"])
