# ============================================================
# CELL 1 — Install dependencies
# ============================================================
# Run this first, then restart the kernel
# ============================================================

!pip install -q transformers accelerate fastapi uvicorn pyngrok bitsandbytes


# ============================================================
# CELL 2 — Load model (Qwen2.5-7B-Instruct with 4-bit quant)
# ============================================================

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"   # swap to "meta-llama/Llama-3.1-8B-Instruct" if preferred

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)

print(f"Loading {MODEL_ID} ...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)
model.eval()
print("Model loaded")
print(f"Device map: {model.hf_device_map}")


# ============================================================
# CELL 3 — Inference function
# ============================================================

def generate_response(messages: list[dict], max_new_tokens: int = 512, temperature: float = 0.1) -> str:
    """
    messages: list of {"role": "system"|"user"|"assistant", "content": "..."}
    Returns the assistant reply as a plain string.
    """
    # Use the model's chat template to format messages
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer([text], return_tensors="pt").to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature if temperature > 0 else None,
            do_sample=temperature > 0,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Strip the input tokens, return only new tokens
    new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


# Quick smoke test
test_reply = generate_response([
    {"role": "system", "content": "You are a helpful assistant. Reply in JSON only."},
    {"role": "user",   "content": '{"thought": "testing", "action": "finish", "action_input": "{\\"answer\\": \\"ok\\"}"}'},
])
print("Smoke test reply:", test_reply[:200])


# ============================================================
# CELL 4 — FastAPI server (runs in background thread)
# ============================================================

import threading
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class GenerateRequest(BaseModel):
    messages:       list[dict]
    max_tokens:     int   = 512
    temperature:    float = 0.1

class GenerateResponse(BaseModel):
    response: str

@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_ID}

@app.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest):
    reply = generate_response(
        req.messages,
        max_new_tokens=req.max_tokens,
        temperature=req.temperature,
    )
    return GenerateResponse(response=reply)

# Start server in a background thread so the notebook stays interactive
def _run_server():
    uvicorn.run(app, host="0.0.0.0", port=8080, log_level="warning")

server_thread = threading.Thread(target=_run_server, daemon=True)
server_thread.start()
print("FastAPI server running on port 8080 ")


# ============================================================
# CELL 5 — Expose via ngrok tunnel
# ============================================================
# Get your free authtoken at https://dashboard.ngrok.com/get-started/your-authtoken
# Paste it below

from pyngrok import ngrok

NGROK_TOKEN = "PASTE_YOUR_NGROK_TOKEN_HERE"   # ← replace this

ngrok.set_auth_token(NGROK_TOKEN)
tunnel = ngrok.connect(8080)
public_url = tunnel.public_url

print("=" * 55)
print(f"  PUBLIC ENDPOINT: {public_url}/generate")
print("=" * 55)
print("Copy the URL above into your local .env:")
print(f"  KAGGLE_LLM_URL={public_url}/generate")


# ============================================================
# CELL 6 — Keep-alive ping (run in last cell, keeps kernel alive)
# ============================================================

import time, requests as req

print("Keep-alive loop running. Stop the cell to shut down.")
while True:
    try:
        r = req.get(f"{public_url}/health", timeout=5)
        print(f"  ping ok — {r.json()}")
    except Exception as e:
        print(f"  ping failed: {e}")
    time.sleep(60)
