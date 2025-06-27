from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import base64
import io
from PIL import Image
import torch
from transformers import AutoProcessor, AutoModelForSeq2SeqLM

app = FastAPI()

# Carica il modello e processor (sostituisci con il modello VL2)
model_name = "deepseek-ai/deepseek-vl2-small"
processor = AutoProcessor.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

class RequestInput(BaseModel):
    text: str
    image_base64: Optional[str] = None  # opzionale

@app.post("/predict")
async def predict(data: RequestInput):
    try:
        inputs = {}
        if data.image_base64:
            # decodifica immagine
            image_bytes = base64.b64decode(data.image_base64)
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            inputs['images'] = image
        
        inputs['text'] = data.text

        # prepara inputs per il modello
        model_inputs = processor(text=inputs.get('text', None),
                                 images=inputs.get('images', None),
                                 return_tensors="pt",
                                 padding=True)

        # inference
        outputs = model.generate(**model_inputs)
        decoded = processor.decode(outputs[0], skip_special_tokens=True)

        return {"result": decoded}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
