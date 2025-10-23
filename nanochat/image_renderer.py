from PIL import Image, ImageDraw, ImageFont

def render_text_to_image(text, image_size=(256, 256), font_size=20):
    """
    Renders a string of text to a Pillow image.
    """
    image = Image.new("RGB", image_size, "white")
    draw = ImageDraw.Draw(image)
    
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except IOError:
        font = ImageFont.load_default()

    draw.text((10, 10), text, fill="black", font=font)

    return image
