from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from PIL import Image
import base64
import requests
from io import BytesIO

from deepseek_vl2.models import DeepseekVLV2Processor, DeepseekVLV2ForCausalLM

app = FastAPI()

# Inizializzazione globale
model_name = "deepseek-ai/deepseek-vl2-small"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

processor = DeepseekVLV2Processor.from_pretrained(model_name, trust_remote_code=True)
model = DeepseekVLV2ForCausalLM.from_pretrained(model_name, trust_remote_code=True).to(device)
model = model.to(torch.bfloat16).eval()

class RequestInput(BaseModel):
    prompt: str
    image: str  # base64 o URL
    max_new_tokens: int = 512

def load_image(image_input: str):
    if image_input.startswith("http"):
        resp = requests.get(image_input)
        img = Image.open(BytesIO(resp.content)).convert("RGB")
    else:
        img = Image.open(BytesIO(base64.b64decode(image_input))).convert("RGB")
    return img

@app.post("/generate")
async def generate(data: RequestInput):
    try:
        img = load_image(data.image)
        text = data.prompt

        conversation = [
            {"role": "<|User|>", "content": f"\n|ref|>{text}<|/ref|>.", "images": [img]},
            {"role": "<|Assistant|>", "content": ""}
        ]
        inputs = processor(conversations=conversation, images=[img], force_batchify=True).to(device)

        inputs_embeds = model.prepare_inputs_embeds(**inputs)

        with torch.no_grad():
            outputs = model.language.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=inputs['attention_mask'],
                pad_token_id=processor.tokenizer.eos_token_id,
                bos_token_id=processor.tokenizer.bos_token_id,
                eos_token_id=processor.tokenizer.eos_token_id,
                max_new_tokens=data.max_new_tokens,
                do_sample=False,
                use_cache=True
            )
        response = processor.tokenizer.decode(outputs[0].tolist(), skip_special_tokens=True)
        return {"response": response}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
