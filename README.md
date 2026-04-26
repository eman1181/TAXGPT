# TaxGPT — مصلحة الضرائب المصرية
> AI-powered Egyptian Tax Law Assistant

---

## 📁 Project Structure

```
TAXGPT/
├── archive/                        ← Legal documents (required)
│   ├── law_2025.docx
│   ├── Health_Contribution.docx
│   ├── income_tax_law.docx
│   ├── instructions_2012.docx
│   └── fatwas.xlsx
├── Voice_TAX/
│   └── voice_full.py
├── GRAD_FINAL_BACKEND.py           ← AI engine (RAG + Qwen LLM)
├── app.py                          ← FastAPI server (port 8000)
├── server.py                       ← Flask server (port 5000)
├── assistant.html                  ← TaxGPT chat interface
├── index.html                      ← Homepage
├── inquiry-form.html               ← Tax inquiry form
├── logo.jpeg
├── requirements.txt
├── start.bat                       ← Windows one-click launcher
└── README.md
```

---

## ⚙️ How It Works

1. User submits a tax question in Arabic via the chat interface
2. The backend retrieves relevant legal text using FAISS vector search
3. The Qwen LLM generates a structured JSON response
4. The response is formatted into an official Arabic tax letter
5. User can edit the letter and download it as a Word (.docx) file

---

## 🖥️ Running on macOS

### Prerequisites
- Python 3.11
- Homebrew (for portaudio if using voice features)

### Step 1 — Clone / Copy the project
Make sure the `archive/` folder contains all `.docx` and `.xlsx` files.

### Step 2 — Create and activate virtual environment
```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

### Step 3 — Install dependencies
```bash
pip install -r requirements.txt
pip install SpeechRecognition
brew install portaudio
pip install pyaudio
```

### Step 4 — Install watchfiles (stops reload spam)
```bash
pip install watchfiles
```

### Step 5 — Run the servers

Open **three separate terminals**, activate `.venv` in each:

**Terminal 1 — FastAPI (main AI server):**
```bash
source .venv/bin/activate
python -m uvicorn app:app --reload --reload-exclude ".venv" --reload-exclude "venv" --port 8000
```

**Terminal 2 — Flask (file upload server):**
```bash
source .venv/bin/activate
python server.py
```

**Terminal 3 — Frontend:**
```bash
python -m http.server 3000
```

### Step 6 — Open the app
Go to: **http://localhost:3000**

Login credentials:
| Email | Password |
|---|---|
| admin@example.com | 1234 |
| user@eta.gov.eg | 5678 |

---

## 🪟 Running on Windows

### Prerequisites
- Python 3.11 → https://www.python.org/downloads/
  - ✅ Make sure to check **"Add Python to PATH"** during installation
- Google Chrome or Microsoft Edge (for correct Arabic display)

### Step 1 — Copy project files
Transfer the entire project folder to the Windows PC. Make sure to include:
- `archive/` folder with all documents
- `requirements.txt`
- All `.py` and `.html` files

❌ Do NOT copy `.venv/`, `venv/`, or `__pycache__/` — these are machine-specific.

### Step 2 — Open Command Prompt in project folder
Right-click the project folder → **Open in Terminal** (or use `cd` to navigate to it)

### Step 3 — Create virtual environment
```cmd
python -m venv venv
venv\Scripts\activate
```

### Step 4 — Install dependencies
```cmd
pip install -r requirements.txt
```
> ⏳ This will take several minutes — it downloads PyTorch, Transformers, and other large packages.

If `torch` fails:
```cmd
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

If `faiss-cpu` fails:
```cmd
pip install faiss-cpu --no-cache-dir
```

### Step 5 — Install voice dependencies
```cmd
pip install SpeechRecognition
pip install pipwin
pipwin install pyaudio
```

### Step 6 — Run with one click
Double-click **`start.bat`** in the project folder.

This will automatically open three terminal windows and launch the browser.

Or run manually in **three separate Command Prompt windows**:

**Terminal 1 — FastAPI:**
```cmd
venv\Scripts\activate
python -m uvicorn app:app --reload --reload-exclude "venv" --port 8000
```

**Terminal 2 — Flask:**
```cmd
venv\Scripts\activate
python server.py
```

**Terminal 3 — Frontend:**
```cmd
python -m http.server 3000
```

### Step 7 — Open the app
Go to: **http://localhost:3000**

---

## 🚀 First Run Notes

- The **first startup takes 3–5 minutes** — the Qwen LLM (~1GB) and sentence transformer (~400MB) models are downloaded automatically from HuggingFace
- After the first run, models are cached and startup is much faster
- Wait until you see this message before testing:
  ```
  FAISS index created with: 110 chunks
  INFO: Application startup complete.
  ```

---

## 📦 Generating requirements.txt (for maintainers)

If you add new packages, regenerate the requirements file:

```bash
# macOS
source .venv/bin/activate
pip freeze > requirements.txt

# Windows
venv\Scripts\activate
pip freeze > requirements.txt
```

---

## 🛠️ Troubleshooting

| Problem | Solution |
|---|---|
| `ModuleNotFoundError: speech_recognition` | Run `pip install SpeechRecognition` |
| `ImportError: cannot import PreTrainedModel` | You're using the wrong Python. Run `python -m uvicorn` not `uvicorn` directly |
| `uvicorn` not found | Run `pip install uvicorn` inside your venv |
| Server keeps reloading in a loop | Run `pip install watchfiles` then restart |
| Port 8000 already in use | Change `--port 8000` to `--port 8001` and update `API_BASE` in `assistant.html` |
| Arabic text displays incorrectly | Use Chrome or Edge — avoid Internet Explorer |
| `pyaudio` install fails on Windows | Use `pipwin install pyaudio` instead of `pip install pyaudio` |
| Frontend can't reach API (CORS error) | Make sure you use `127.0.0.1` or `localhost` consistently in both frontend and backend |
| Model download is slow | First run only — wait for `Application startup complete` |

---

## 🔑 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/ping` | Health check |
| POST | `/tax-answer` | Generate tax letter from question |
| POST | `/export-docx` | Export letter as Word file |
| POST | `/feedback` | Save user rating and edits |

Test the API is running:
```
http://localhost:8000/ping
→ {"status": "ok"}
```

---

