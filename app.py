# Fast-API
from fastapi import FastAPI, Request
from pydantic import BaseModel
# from transformers import T5ForConditionalGeneration, T5Tokenizer
from transformers import T5ForConditionalGeneration, AutoTokenizer
import torch
import re
from fastapi.templating import Jinja2Templates  #UI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles #Images and CSS
from fastapi import Form

#Initialize our fastapi app
app = FastAPI(title="Text summarizor app", description="Text summarization using T5", version="1.0")

# load our model and tokenizer from hugging face
model_name = "Ishikabharadwaj/text-summarizer-model"

model = T5ForConditionalGeneration.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# device

# Check whether GPU is available or not
# if torch.cuda.is_available():
#     device = torch.device("cuda")   # Use NVIDIA GPU
#     print("GPU is available!")
#     print("GPU Name:", torch.cuda.get_device_name(0))
# else:
#     device = torch.device("cpu")    # Fallback to CPU
#     print("GPU not available, using CPU")

# # Move model to selected device (GPU or CPU)
# model.to(device)

device = torch.device("cpu")
print("Using device:", device)

# templating
templates = Jinja2Templates(directory=".")

# Input Schema for dialogue => string
class DialogueInput(BaseModel):
    dialogue: str

def clean_data(text):
  text = re.sub(r"\r\n"," ", text) # lines
  text = re.sub(r"\r\s+"," ", text) # spaces
  text = re.sub(r"\r<.*?>"," ", text) # html tag
  text = text.strip().lower()
  return text

def summarize_diaglogue(dialogue:str) -> str:
  dialogue = clean_data(dialogue) # clean

  # tokenize
  inputs = tokenizer(
      dialogue,
      max_length=512,
      padding="max_length",
      truncation=True,
      return_tensors="pt"
  )

  # generate the summary => token ids
#   model.to(device)
  summary_ids = model.generate(
      inputs["input_ids"],
      attention_mask = inputs["attention_mask"],
      max_length=150,
      num_beams=4,
      early_stopping=True
  )

  # token ids convert to summary => decoding ( decode our output)
  summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

  return summary    


  # API Endpoints
from fastapi import Form

@app.post("/summarize", response_class=HTMLResponse)
async def create_summary(request: Request, dialogue: str = Form(...)):

    summary = summarize_diaglogue(dialogue)

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "summary": summary
        }
    )

@app.get("/", response_class=HTMLResponse)
async def home(request:Request):
    return templates.TemplateResponse(request=request, name="index.html")
