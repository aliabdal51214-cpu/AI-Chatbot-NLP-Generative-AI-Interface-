<img width="3085" height="1492" alt="Screenshot 2" src="https://github.com/user-attachments/assets/caa1717e-ceb8-4dcb-b6fe-fe5c66801c54" />

<img width="3069" height="1505" alt="Screenshot 1" src="https://github.com/user-attachments/assets/5191aecd-64f7-424b-b601-cfb19d46ddba" />
# 🤖 AI Chatbot — NLP & Generative AI Interface

![Python](https://img.shields.io/badge/Python-3.10-blue?style=for-the-badge&logo=python&logoColor=white)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow?style=for-the-badge&logo=huggingface&logoColor=white)
![Gradio](https://img.shields.io/badge/Gradio-UI-orange?style=for-the-badge&logo=gradio&logoColor=white)
![DialoGPT](https://img.shields.io/badge/Model-DialoGPT--Medium-purple?style=for-the-badge&logo=openai&logoColor=white)
![Colab](https://img.shields.io/badge/Google%20Colab-Ready-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white)

---

## 📌 Project Overview

A **conversational AI chatbot** powered by Microsoft's **DialoGPT-Medium** model, built with **Hugging Face Transformers** and deployed via a **Gradio web interface**. The chatbot maintains multi-turn conversation history, generates contextually aware responses, and is accessible via a public shareable link — no server setup required.

> Built as part of my NLP & Generative AI learning journey — from zero to deployed chatbot.

---

## 🚀 Live Demo

> Run instantly on Google Colab — no GPU needed!

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)

---

## 🧠 Tech Stack

| Layer | Technology |
|---|---|
| **Language Model** | Microsoft DialoGPT-Medium |
| **NLP Framework** | Hugging Face Transformers |
| **UI Framework** | Gradio |
| **Deep Learning** | PyTorch |
| **Runtime** | Google Colab / Python 3.10 |
| **Deployment** | Gradio Live Share Link |

---

## ✨ Features

- 💬 **Multi-turn conversations** — remembers full chat history per session
- 🧠 **DialoGPT-Medium** — 345M parameter conversational model by Microsoft
- 🌐 **Public shareable link** — instantly share via Gradio's `share=True`
- 🔄 **Clear & Reset** — wipe conversation memory with one click
- ⚡ **Zero setup** — runs entirely in Google Colab, no local install needed
- 🎛️ **Sampling controls** — `top_k`, `top_p`, `temperature` for natural responses

---

## 📁 Project Structure

```
ai-chatbot-dialogpt/
│
├── chatbot.py          # Main application code
├── requirements.txt    # Python dependencies
└── README.md           # Project documentation
```

---

## ⚙️ Installation & Usage

### Option 1 — Google Colab (Recommended)

Paste the full code into a single Colab cell and run:

```python
!pip install transformers gradio torch -q
```

The rest of the code handles model loading and UI launch automatically.

### Option 2 — Local Setup

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/ai-chatbot-dialogpt.git
cd ai-chatbot-dialogpt

# Install dependencies
pip install -r requirements.txt

# Run the app
python chatbot.py
```

---

## 🧾 requirements.txt

```
transformers
gradio
torch
```

---

## 💻 Full Source Code

```python
# Install required libraries
!pip install transformers gradio torch -q

import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load model and tokenizer
print("Loading model... please wait")
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")
print("Model loaded successfully!")

# Chat history
chat_history_ids = None

def chat(user_input):
    global chat_history_ids

    new_input_ids = tokenizer.encode(
        user_input + tokenizer.eos_token,
        return_tensors="pt"
    )

    bot_input_ids = (
        torch.cat([chat_history_ids, new_input_ids], dim=-1)
        if chat_history_ids is not None
        else new_input_ids
    )

    chat_history_ids = model.generate(
        bot_input_ids,
        max_length=1000,
        pad_token_id=tokenizer.eos_token_id,
        no_repeat_ngram_size=3,
        do_sample=True,
        top_k=100,
        top_p=0.7,
        temperature=0.8
    )

    response = tokenizer.decode(
        chat_history_ids[:, bot_input_ids.shape[-1]:][0],
        skip_special_tokens=True
    )
    return response

def reset():
    global chat_history_ids
    chat_history_ids = None

# Gradio UI
with gr.Blocks(title="AI Chatbot (DialoGPT)") as demo:
    gr.Markdown("# AI Chatbot (DialoGPT)")
    gr.Markdown("A simple conversational AI chatbot built using Hugging Face + Gradio")

    chatbot = gr.Chatbot(label="Conversation")
    user_input = gr.Textbox(placeholder="Type your message here...", label="Your Message")

    with gr.Row():
        submit_btn = gr.Button("Submit", variant="primary")
        clear_btn = gr.Button("Clear")

    history = gr.State([])

    def respond(message, chat_history):
        bot_reply = chat(message)
        chat_history.append((message, bot_reply))
        return "", chat_history

    def clear_chat():
        reset()
        return [], []

    submit_btn.click(respond, [user_input, history], [user_input, chatbot])
    user_input.submit(respond, [user_input, history], [user_input, chatbot])
    clear_btn.click(clear_chat, [], [chatbot, history])

demo.launch(share=True)
```

---

## 🧪 How It Works

```
User Input
    │
    ▼
Tokenizer encodes text + EOS token
    │
    ▼
Appended to conversation history tensor
    │
    ▼
DialoGPT-Medium generates response
(top_k=100, top_p=0.7, temperature=0.8)
    │
    ▼
Decoded response displayed in Gradio UI
```

---

## 📊 Model Details

| Property | Value |
|---|---|
| Model | `microsoft/DialoGPT-medium` |
| Parameters | 345 Million |
| Training Data | 147M Reddit conversations |
| Max Length | 1000 tokens |
| Sampling | top_k=100, top_p=0.7, temp=0.8 |

---

## 🎯 Skills Demonstrated

- ✅ Natural Language Processing (NLP)
- ✅ Generative AI / Large Language Models
- ✅ Hugging Face Transformers API
- ✅ PyTorch tensor operations
- ✅ Gradio UI development
- ✅ Model deployment & sharing
- ✅ Conversational AI / dialogue systems

---

## 🔮 Future Improvements

- [ ] Swap DialoGPT for GPT-2 or Llama for better responses
- [ ] Add memory summarization for long conversations
- [ ] Deploy permanently on Hugging Face Spaces
- [ ] Add user authentication
- [ ] Support multiple languages

---

## 👤 Author 

**ALI ABDAL**
- 🔗 LinkedIn: https://www.linkedin.com/in/ali-abdal-ml/
- 🐙 GitHub: https://github.com/aliabdal51214-cpu

---

## 📄 License

This project is open source under the [MIT License](LICENSE).

---

> ⭐ If you found this useful, please give it a star on GitHub!
