# create_header.py
from PIL import Image, ImageDraw, ImageFont
import os
os.makedirs("assets", exist_ok=True)
img_w, img_h = 1200, 200
img = Image.new("RGBA", (img_w, img_h), (11,11,11,255))
draw = ImageDraw.Draw(img)
# gold gradient
for i in range(img_w):
    r = int(40 + (255-40)*(i/img_w))
    g = int(30 + (215-30)*(i/img_w))
    b = int(20 + (140-20)*(i/img_w))
    draw.line([(i,0),(i,img_h)], fill=(r,g,b,255))
# dark overlay
overlay = Image.new("RGBA", (img_w, img_h), (0,0,0,110))
img = Image.alpha_composite(img, overlay)
draw = ImageDraw.Draw(img)
try:
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
    title_font = ImageFont.truetype(font_path, 64)
    sub_font = ImageFont.truetype(font_path, 20)
except Exception:
    title_font = ImageFont.load_default()
    sub_font = ImageFont.load_default()
draw.text((60,50), "Golden Market Analyzer", font=title_font, fill=(255,245,180,255))
draw.text((60,130), "Live prices · Indicators · Signals · News", font=sub_font, fill=(230,230,230,200))
img.convert("RGB").save("assets/header.png", "PNG")
print("Saved assets/header.png")
