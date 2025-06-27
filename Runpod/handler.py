from transformers import AutoTokenizer, AutoModelForVision2Seq
import torch
from PIL import Image
import base64
import io
import os

# Load model once (cold start)
model_id = "deepseek-ai/deepseek-vl-2-small"
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForVision2Seq.from_pretrained(model_id, torch_dtype=torch.float16, trust_remote_code=True).cuda()
model.eval()

def decode_image(image_b64):
    image_bytes = base64.b64decode(image_b64)
    return Image.open(io.BytesIO(image_bytes)).convert("RGB")

def handler(event):
    prompt = event["input"].get("prompt", "")
    image_b64 = event["input"].get("image", None)

    if image_b64:
        image = decode_image(image_b64)
        prompt = "<image>\n" + prompt
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        inputs.update({"pixel_values": tokenizer.image_processor(image, return_tensors="pt").pixel_values.half().to("cuda")})
    else:
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=256)

    decoded = tokenizer.decode(output[0], skip_special_tokens=True)
    return {"output": decoded}
