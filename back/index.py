# The model you want to use
model = "CompVis/stable-diffusion-v1-4"
# Steps. Basically the quality of the image. More steps, the longer it takes but the quality is better
steps = 50
# Resolution (Y,X)
resolution = [256,256]

from fastapi import FastAPI, Query
from diffusers import StableDiffusionPipeline
import torch
import uuid
import threading
import time
import os
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from fastapi import FastAPI

@asynccontextmanager
async def lifespan(app: FastAPI):
    threading.Thread(target=generate_loop, daemon=True).start()
    yield

app = FastAPI(lifespan=lifespan)

from fastapi.staticfiles import StaticFiles
app.mount("/output", StaticFiles(directory="output"), name="output")

pipe = StableDiffusionPipeline.from_pretrained(model)
pipe = pipe.to("cuda")
pipe.safety_checker = None

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

current_prompt = ""
stop_event = threading.Event()

os.makedirs("output", exist_ok=True)

def generate_loop():
    while not stop_event.is_set():
        if current_prompt.strip():
            image = pipe(current_prompt, num_inference_steps=steps, height=resolution[0], width=resolution[1]).images[0]
            filename = f"output/{uuid.uuid4().hex}.png"
            image.save(filename)
        time.sleep(1)

@app.get("/update_prompt")
def update_prompt(prompt: str = Query(...)):
    global current_prompt
    current_prompt = prompt
    return {"status": "prompt updated", "prompt": prompt}

@app.get("/images")
def get_images():
    files = sorted(os.listdir("output"), reverse=True)[0:10]
    return {"images": files}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
