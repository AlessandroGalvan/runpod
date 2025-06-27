import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq
import base64
import requests
from io import BytesIO
import json
from fastapi import FastAPI, Request
import uvicorn

app = FastAPI()

# Load model and processor
model_id = "deepseek-ai/deepseek-vl-2-small"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForVision2Seq.from_pretrained(model_id, trust_remote_code=True).to(device)
model.eval()

def load_image(image_input):
    if image_input.startswith("http://") or image_input.startswith("https://"):
        response = requests.get(image_input)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(BytesIO(base64.b64decode(image_input))).convert("RGB")
    return image

@app.post("/generate")
async def generate(request: Request):
    data = await request.json()
    prompt = data.get("prompt")
    image_input = data.get("image")

    if not prompt or not image_input:
        return {"error": "Both 'prompt' and 'image' fields are required."}

    try:
        image = load_image(image_input)
        inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=512)
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return {"response": generated_text}
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
