# app_fastapi.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
import torch
from fastapi.middleware.cors import CORSMiddleware
import logging

# ---------- Configuration ----------
MODEL_DIR = "models/mt_en_de_pipeline/checkpoint-8750"
DEVICE = "cpu"  # CPU-only
BATCH_SIZE = 4
DEFAULT_NUM_BEAMS = 2

# ---------- Logging ----------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------- FastAPI setup ----------
app = FastAPI(title="MT API (EN<->DE)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class TranslateRequest(BaseModel):
    texts: List[str]
    src_lang: Optional[str] = "en"
    tgt_lang: Optional[str] = "de"
    num_beams: Optional[int] = None
    max_length: Optional[int] = 128

# ---------- Load model & tokenizer ----------
logger.info(f"Loading tokenizer and model from: {MODEL_DIR}")
try:
    tokenizer = M2M100Tokenizer.from_pretrained(MODEL_DIR)
    model = M2M100ForConditionalGeneration.from_pretrained(MODEL_DIR)
    model.to(DEVICE)
    model.eval()
    logger.info(f"Model loaded successfully on {DEVICE}")
except Exception as e:
    logger.exception("Failed to load model. Check MODEL_DIR and installed packages.")
    raise RuntimeError("Model load failed") from e

@app.get("/")
def root():
    return {"status": "ok", "device": DEVICE, "model_dir": MODEL_DIR}

@app.post("/translate")
def translate(req: TranslateRequest):
    texts = req.texts
    if not texts:
        raise HTTPException(status_code=400, detail="Empty texts list.")

    tokenizer.src_lang = req.src_lang
    tokenizer.tgt_lang = req.tgt_lang

    outputs = []
    num_beams = req.num_beams if req.num_beams else DEFAULT_NUM_BEAMS

    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i:i+BATCH_SIZE]
        enc = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=req.max_length)
        enc = {k: v.to(DEVICE) for k, v in enc.items()}
        forced_bos = tokenizer.get_lang_id(req.tgt_lang)

        with torch.no_grad():
            gen = model.generate(
                **enc,
                forced_bos_token_id=forced_bos,
                num_beams=num_beams,
                max_length=req.max_length,
                early_stopping=True,
            )
        decoded = tokenizer.batch_decode(gen, skip_special_tokens=True)
        outputs.extend(decoded)

    return {"translations": outputs}
