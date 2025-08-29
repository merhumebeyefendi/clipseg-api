from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline
from PIL import Image
import io, base64, numpy as np

app = FastAPI()
pipe = pipeline("zero-shot-image-segmentation", model="CIDAS/clipseg-rd64-refined")

class Req(BaseModel):
    image: str
    text: str
    threshold: float = 0.4

def decode_image(s: str):
    b64 = s.split("base64,")[-1]
    return Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")

def to_data_url_png(img):
    buf = io.BytesIO(); img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

@app.get("/healthz")
def health(): return {"ok": True}

@app.post("/predict")
def predict(req: Req):
    img = decode_image(req.image)
    out = pipe(image=img, text=req.text)
    if not out: return {"mask": None}
    m = out[0]["mask"]
    if isinstance(m, Image.Image):
        mask = m
    else:
        arr = np.array(m)
        vmax = float(arr.max()) if float(arr.max()) > 0 else 1.0
        arr = (arr / vmax) >= req.threshold
        mask = Image.fromarray((arr.astype(np.uint8) * 255))
    return {"mask": to_data_url_png(mask)}
