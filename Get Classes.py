"""
Run this once inside the adversight/ folder to generate imagenet_classes.txt
    python get_classes.py
"""
import urllib.request, json

url = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
with urllib.request.urlopen(url) as r:
    labels = json.load(r)

with open("imagenet_classes.txt", "w") as f:
    for label in labels:
        f.write(label + "\n")

print(f"Done. {len(labels)} classes written to imagenet_classes.txt")