import gradio as gr
from transformers import T5Tokenizer, T5ForConditionalGeneration

# Load model from Hugging Face Hub
model_name = "Ishikabharadwaj/text-summarizer-model"

tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# Summarization function
def summarize_text(text):

    inputs = tokenizer.encode(
        "summarize: " + text,
        return_tensors="pt",
        max_length=512,
        truncation=True
    )

    summary_ids = model.generate(
        inputs,
        max_length=80,
        min_length=20,
        length_penalty=2.0,
        num_beams=4,
        early_stopping=True
    )

    summary = tokenizer.decode(
        summary_ids[0],
        skip_special_tokens=True
    )

    return summary

# Gradio UI
interface = gr.Interface(
    fn=summarize_text,
    inputs=gr.Textbox(
        lines=10,
        placeholder="Enter long text here..."
    ),
    outputs="text",
    title="AI Text Summarizer",
    description="Summarize long text using Transformer-based AI model."
)

interface.launch(server_name="0.0.0.0", server_port=7860)
