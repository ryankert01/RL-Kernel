import sys
from PIL import Image
im = Image.open(sys.argv[1]); h = int(sys.argv[3]) if len(sys.argv) > 3 else 1500
prefix = sys.argv[2]
n = 0
for y in range(0, im.height, h):
    im.crop((0, y, im.width, min(y + h, im.height))).save(f"{prefix}_{n:02d}.png"); n += 1
print(im.size, n, "parts")
