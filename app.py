from transformers import BartForConditionalGeneration, BartTokenizer
from transformers import pipeline
import gradio as gr

# Load summarizer 
tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")

# Load classifier
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

labels = ["legal", "medical", "financial", "general"]

def analyze_document(file):
    if file.name.endswith(".pdf"):
        import fitz
        doc = fitz.open(file.name)
        text = ""
        for page in doc:
            text += page.get_text()
    else:
        with open(file.name, "r", encoding="utf-8") as f:
            text = f.read()

    # Safety check
    if not text.strip():
        return "Could not extract text from file.", "unknown"

    # Summarization
    inputs = tokenizer(text[:1024], return_tensors="pt", truncation=True, max_length=1024)
    summary_ids = model.generate(inputs["input_ids"], max_new_tokens=200)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    # Classification
    classification = classifier(text[:1000], candidate_labels=labels)

    return summary, classification["labels"][0]

demo = gr.Interface(
    fn=analyze_document,
    inputs=gr.File(label="Upload Your Document Here"),
    outputs=[gr.Textbox(label="Summary"), gr.Textbox(label="Document Type")],
    title="Document Classifier and Summarizer"
)

demo.launch(share=True)
