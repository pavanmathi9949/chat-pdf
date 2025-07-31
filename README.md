# ChatPDF

A local PDF‑chat application leveraging **TinyLlama (1.1B via Ollama)**, **LangChain**, and **Streamlit**.

---

## 🚀 Getting Started

### Prerequisites
- Python 3.8+  
- Ollama CLI installed and running  
- Virtual environment support (e.g. `venv`)

### Installation

```bash
# Clone the repository
git clone https://github.com/pavanmathi9949/chat-pdf.git
cd chat-pdf

# Pull the TinyLlama 1.1B model via Ollama
ollama pull tinyllama:1.1b

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate     # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## 🏁 Running the Application

### Backend
```bash
python app.py
```

### Streamlit frontend UI
```bash
streamlit run app_ui.py
```

Then open the URL shown in your terminal (typically `http://localhost:8501`) to start chatting with your PDF.

---

## ⚙️ Environment Variables

Add the following to your `.env`:

```env
OLLAMA_MODEL="tinyllama:1.1b"
```

This configures the application to use TinyLlama (1.1 B parameters) via Ollama. TinyLlama is compact (~638 MB with Q4_0 quantization), resource-efficient, and designed for environments with limited compute :contentReference[oaicite:1]{index=1}.

---

## 🎯 Features

- Upload and embed PDF content for RAG‑based Q&A  
- Use **TinyLlama 1.1B** as the local LLM through Ollama  
- Interactive **Streamlit** chat interface  
- Separate backend (`app.py`) and frontend (`app_ui.py`)  


---

## 🧠 Architecture

- **Backend** (`app.py`): handles PDF ingestion, chunking, embeddings, retrieval  
- **Frontend** (`app_ui.py`): Streamlit chatbot interface  
- **Local LLM**: TinyLlama 1.1B accessed via Ollama  
- **Vector Store**: ChromaDB or equivalent for semantic search  

---

## 📝 How It Works

1. Pull **TinyLlama 1.1B** model: `ollama pull tinyllama:1.1b`  
2. Set up Python environment and install dependencies  
3. Run backend: `python app.py`  
4. Launch UI: `streamlit run app_ui.py`  
5. In Streamlit UI, upload a PDF and wait for embeddings  
6. Choose TinyLlama (or other model, if configured), and send queries  
7. Responses are generated using retrieval-augmented generation over embedded chunks  

---

## ⚠️ TinyLlama Overview & Limitations

TinyLlama is a 1.1 B parameter model pretrained on **3 trillion tokens** over approximately 90 days using 16 A100‑40G GPUs :contentReference[oaicite:2]{index=2}. It shares architecture and tokenizer with Llama 2, enabling seamless integration with Llama-based projects :contentReference[oaicite:3]{index=3}.

Its compact size makes it ideal for limited-resource environments, but this also means:
- It may struggle with complex reasoning or nuanced queries.
- Outputs can be inconsistent or low-quality compared to larger models.
- Benchmarks show average performance (~52.99 average on MMLU, HellaSwag, WinoGrande) — better than older ~1 B models but below stronger 7B+ models :contentReference[oaicite:4]{index=4}.

For higher fidelity, consider larger models (e.g. Llama 2 7B+ or Gemma 3+).

---

## 🛠️ Troubleshooting

- **Model missing?** Make sure `ollama pull tinyllama:1.1b` succeeds.  
- **Virtual env errors?** On Windows, run `venv\Scripts\activate`.  
- **Streamlit issues?** Ensure `streamlit` is included in `requirements.txt`.  
- **PDF errors?** Avoid encrypted or corrupted files; test with simple PDFs first.  
- **Performance slow?** Large PDFs can take time for embedding—monitor logs for progress.

---

## 📂 Recommended Structure

```
chat-pdf/
├── app.py            # Backend logic
├── app_ui.py         # Streamlit UI frontend
├── requirements.txt
├── .env.example      # Sample environment variables
├── vector_store/     # Generated embeddings (e.g. Chroma)
└── README.md
```

---

## ✅ Summary

| Task                            | Command / Setting                            |
|---------------------------------|----------------------------------------------|
| Set default model               | `OLLAMA_MODEL="tinyllama:1.1b"`              |
| Pull the model via Ollama       | `ollama pull tinyllama:1.1b`                |
| Setup virtual environment       | `python -m venv venv && source venv/bin/activate` |
| Install dependencies            | `pip install -r requirements.txt`           |
| Run backend server              | `python app.py`                              |
| Launch Streamlit UI             | `streamlit run app_ui.py`                    |

---

## 🤝 Contributing

Contributions and bug reports welcome! To contribute:
1. Fork the repo  
2. Create a feature branch (`git checkout -b feature-name`)  
3. Commit your changes (`git commit -m "Add feature"`)  
4. Push and open a Pull Request with a clear description  

Please follow existing project conventions and update tests or documentation where applicable.

---

## 📜 License

This project is MIT‑licensed. See the LICENSE file for details.

---

## 🧾 Credits

Developed by **pavanmathi9949**, using **TinyLlama 1.1B**, **LangChain**, and **Streamlit**.  
TinyLlama was trained by the StatNLP group at SUTD, sharing architecture and tokenizer with Llama 2 and trained on 3 trillion tokens :contentReference[oaicite:5]{index=5}.
