from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI
import base64

client = OpenAI()

response = client.images.generate(
    model="gpt-image-1",
    prompt="Cyberpunk owl face, geometric wireframe style, glowing gold eyes, dark background, neon gold and teal accents, digital circuit patterns, minimalist, no text",
    size="1024x1024",
    n=1,
)

image_data = base64.b64decode(response.data[0].b64_json)
with open("owl.png", "wb") as f:
    f.write(image_data)

print("Saved as owl.png")