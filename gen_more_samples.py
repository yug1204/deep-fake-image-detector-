from PIL import Image, ImageDraw, ImageFilter, ImageOps
import random
import os
import math

def generate_samples():
    if not os.path.exists("samples"):
        os.makedirs("samples")
    
    # --- GENERATE REAL SAMPLES (Natural patterns, smooth gradients) ---
    print("Generating Real Samples...")
    
    # 1. Real: Portrait (Simple Gradient Face)
    img = Image.new('RGB', (160, 160), color=(255, 220, 177))
    draw = ImageDraw.Draw(img)
    draw.ellipse((40, 40, 120, 140), fill=(210, 180, 140)) # Face
    draw.ellipse((55, 70, 75, 80), fill=(50, 50, 50)) # Eye L
    draw.ellipse((95, 70, 115, 80), fill=(50, 50, 50)) # Eye R
    img.save("samples/real_portrait_01.png")
    
    # 2. Real: Landscape (Blue Sky, Green Grass)
    img = Image.new('RGB', (160, 160), color=(135, 206, 235))
    draw = ImageDraw.Draw(img)
    draw.rectangle((0, 100, 160, 160), fill=(34, 139, 34))
    draw.ellipse((120, 10, 150, 40), fill=(255, 255, 0)) # Sun
    img.save("samples/real_landscape_02.png")
    
    # 3. Real: Macro (Flower pattern)
    img = Image.new('RGB', (160, 160), color=(200, 255, 200))
    draw = ImageDraw.Draw(img)
    for i in range(0, 360, 45):
        draw.line((80, 80, 80 + 40*math.cos(math.radians(i)), 80 + 40*math.sin(math.radians(i))), fill=(255, 105, 180), width=10)
    draw.ellipse((70, 70, 90, 90), fill=(255, 255, 0))
    img.save("samples/real_flower_03.png")

    # 4. Real: Night City (Dark with bright spots)
    img = Image.new('RGB', (160, 160), color=(10, 10, 20))
    draw = ImageDraw.Draw(img)
    for _ in range(20):
        x, y = random.randint(0, 160), random.randint(0, 160)
        draw.rectangle((x, y, x+5, y+10), fill=(255, 255, 200))
    img.save("samples/real_night_04.png")

    # 5. Real: Texture (Wood/Sand noise)
    img = Image.new('RGB', (160, 160), color=(194, 178, 128))
    draw = ImageDraw.Draw(img)
    for _ in range(1000):
        x, y = random.randint(0, 160), random.randint(0, 160)
        draw.point((x, y), fill=(160, 82, 45))
    img.save("samples/real_texture_05.png")

    # --- GENERATE FAKE SAMPLES (Artifacts, Glitches, Unnatural) ---
    print("Generating Fake Samples...")

    # 1. Fake: Glitch (RGB Shift)
    img = Image.new('RGB', (160, 160), color=(100, 100, 100))
    draw = ImageDraw.Draw(img)
    draw.rectangle((40, 40, 120, 120), fill=(200, 200, 200))
    r, g, b = img.split()
    r = ImageOps.crop(r, border=2)
    r = r.resize((160, 160))
    img = Image.merge('RGB', (r, g, b)) # Slight shift
    img.save("samples/deepfake_glitch_01.png")

    # 2. Fake: Grid (GAN Checkerboard artifact)
    img = Image.new('RGB', (160, 160), color=(50, 50, 50))
    draw = ImageDraw.Draw(img)
    for i in range(0, 160, 10):
        for j in range(0, 160, 10):
            if (i+j) % 20 == 0:
                draw.rectangle((i, j, i+10, j+10), fill=(60, 60, 60))
    img.save("samples/deepfake_grid_02.png")

    # 3. Fake: Blur (Loss of detail)
    img = Image.new('RGB', (160, 160), color=(100, 50, 50))
    draw = ImageDraw.Draw(img)
    draw.rectangle((50, 50, 110, 110), fill=(200, 100, 100))
    img = img.filter(ImageFilter.GaussianBlur(radius=4))
    img.save("samples/deepfake_blur_03.png")

    # 4. Fake: Noise (High frequency noise)
    img = Image.new('RGB', (160, 160), color=(20, 20, 20))
    draw = ImageDraw.Draw(img)
    for _ in range(5000):
        x, y = random.randint(0, 160), random.randint(0, 160)
        draw.point((x, y), fill=(0, 255, 0)) # Digital matrix noise
    img.save("samples/deepfake_noise_04.png")

    # 5. Fake: Distortion (Warped)
    img = Image.new('RGB', (160, 160), color=(100, 100, 200))
    draw = ImageDraw.Draw(img)
    draw.ellipse((40, 40, 120, 120), outline=(255, 255, 255), width=5)
    # Simple logic to just create a weird pattern
    for y in range(160):
        for x in range(160):
            if (x * y) % 20 < 2:
                img.putpixel((x, y), (255, 0, 0))
    img.save("samples/deepfake_warp_05.png")

    print("All 10 samples generated in 'samples/' folder.")

if __name__ == "__main__":
    generate_samples()
