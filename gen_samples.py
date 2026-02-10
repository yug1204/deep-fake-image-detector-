from PIL import Image, ImageDraw, ImageFilter
import random
import os

def create_real_sample():
    # Create a "Natural" looking image (Smooth gradient + soft shapes)
    img = Image.new('RGB', (300, 300), color=(135, 206, 235))
    draw = ImageDraw.Draw(img)
    # Draw a "Sun"
    draw.ellipse((200, 50, 280, 130), fill=(255, 255, 0))
    # Draw "Grass"
    draw.rectangle((0, 250, 300, 300), fill=(34, 139, 34))
    # Save
    img.save("sample_real_person.png")

def create_fake_sample():
    # Create an "Unnatural" looking image (Noise, artifacts, weird colors)
    img = Image.new('RGB', (300, 300), color=(50, 50, 50))
    draw = ImageDraw.Draw(img)
    
    # Random Noise artifacts
    for _ in range(500):
        x = random.randint(0, 300)
        y = random.randint(0, 300)
        draw.point((x, y), fill=(random.randint(0,255), 0, 0))
    
    # Weird geometric shapes (FFT artifacts)
    draw.rectangle((100, 100, 200, 200), outline=(0, 255, 0))
    draw.line((0,0, 300,300), fill=(255, 0, 0), width=2)
    
    img.save("sample_deepfake_gen.png")

if __name__ == "__main__":
    create_real_sample()
    create_fake_sample()
    print("Samples created.")
