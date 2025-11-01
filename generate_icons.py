"""
Generate placeholder PWA icons
Run this once to create icon files
"""
from PIL import Image, ImageDraw, ImageFont
import os

def create_icon(size, filename):
    """Create a simple app icon"""
    
    # Create directory if needed
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    # Create image with gradient background
    img = Image.new('RGB', (size, size), color=(26, 29, 46))
    draw = ImageDraw.Draw(img)
    
    # Draw circle
    padding = size // 8
    draw.ellipse([padding, padding, size-padding, size-padding], 
                 fill=(52, 152, 219), outline=(41, 128, 185), width=size//40)
    
    # Draw shield icon (simplified)
    shield_points = [
        (size//2, padding + size//8),  # Top
        (size - padding*2, padding + size//4),  # Right
        (size - padding*2, size - padding*2),  # Bottom right
        (size//2, size - padding),  # Bottom
        (padding*2, size - padding*2),  # Bottom left
        (padding*2, padding + size//4),  # Left
    ]
    draw.polygon(shield_points, fill=(255, 255, 255), outline=(230, 230, 230))
    
    # Save
    img.save(filename)
    print(f"[SUCCESS] Created icon: {filename}")

def create_placeholder_image():
    """Create placeholder image for when no video"""
    img = Image.new('RGB', (640, 360), color=(26, 29, 46))
    draw = ImageDraw.Draw(img)
    
    # Draw centered text
    text = "No Video Feed"
    try:
        font = ImageFont.truetype("arial.ttf", 40)
    except:
        font = ImageFont.load_default()
    
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    
    x = (640 - text_width) // 2
    y = (360 - text_height) // 2
    
    draw.text((x, y), text, fill=(149, 165, 166), font=font)
    
    img.save('static/placeholder.jpg', quality=85)
    print("[SUCCESS] Created placeholder image: static/placeholder.jpg")

if __name__ == '__main__':
    # Create icons
    create_icon(192, 'static/icon-192.png')
    create_icon(512, 'static/icon-512.png')
    create_placeholder_image()
    print("\n✅ All icons and placeholder created!")
