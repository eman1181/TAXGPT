#!/usr/bin/env python
# coding: utf-8

# In[2]:


#get_ipython().system('pip install sentence-transformers faiss-cpu python-docx pandas docx2txt')
#get_ipython().system('pip install transformers accelerate sentencepiece')
#get_ipython().system('pip install -U transformers accelerate sentencepiece bitsandbytes einops')


# In[3]:
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "archive"


import torch
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))


# In[4]:


from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import docx2txt
import pandas as pd
import os


# In[5]:


def load_docx(path):
    try:
        text = docx2txt.process(path)
        return text
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return ""


# Load all your provided files
docs = {}

docs["law_2025"] = load_docx(DATA_DIR / "law_2025.docx")
docs["health_contribution"] = load_docx(DATA_DIR / "Health_Contribution.docx")
docs["income_tax"] = load_docx(DATA_DIR / "income_tax_law.docx")
docs["instructions_2012"] = load_docx(DATA_DIR / "instructions_2012.docx")

for name, content in docs.items():
    print(f"{name}: {len(content)} characters loaded.")


# In[6]:


def chunk_text(text, max_length=500, overlap=50):
    words = text.split()
    chunks = []
    i = 0

    while i < len(words):
        chunk = words[i:i + max_length]
        chunks.append(" ".join(chunk))
        i += max_length - overlap

    return chunks

all_chunks = []

for name, content in docs.items():
    chunks = chunk_text(content)
    for c in chunks:
        all_chunks.append((name, c))  # store (source, text)


len(all_chunks)


# In[7]:


# Load excel file
df_fatwas = pd.read_excel(DATA_DIR / "fatwas.xlsx")
df_fatwas = df_fatwas.dropna(how="all")

# Assign the correct columns manually
fatwa_q_col = "الاستفسار"
fatwa_a_col = 'الفتوى الضريبية " قيمة مضافة"  " الرد على الاستفسار "'

print("Detected question column:", fatwa_q_col)
print("Detected answer column:", fatwa_a_col)

# Add fatwa Q&A as chunks
for idx, row in df_fatwas.iterrows():
    q = f"FATWA_QUESTION: {row[fatwa_q_col]}"
    a = f"FATWA_ANSWER: {row[fatwa_a_col]}"
    all_chunks.append(("fatwa", q))
    all_chunks.append(("fatwa", a))


# In[8]:


embedding_model = SentenceTransformer('all-mpnet-base-v2')


texts = [chunk[1] for chunk in all_chunks]
embeddings = embedding_model.encode(texts, convert_to_numpy=True, show_progress_bar=True)

embeddings.shape


# In[9]:


dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

print("FAISS index created with:", index.ntotal, "chunks")


# In[10]:


def retrieve(query, top_k=5):
    query_emb = embedding_model.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_emb, top_k)

    results = []
    for i in indices[0]:
        source, text = all_chunks[i]
        results.append({"source": source, "text": text})
    return results


# Test:
#retrieve("What is Article 40 about?")
#retrieve("فتوى ضريبية عن خصم تحت حساب الضريبة")
#len(all_chunks)


# In[11]:


from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import json
import re
from textwrap import dedent


# In[12]:


import torch  

model_name = "Qwen/Qwen2.5-0.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

llm_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    trust_remote_code=True,
    torch_dtype=torch.float32,   # CPU عادي
).to("cpu")



# In[13]:


#import re, json
#from textwrap import dedent

def build_reasoning_prompt(question, retrieved_chunks, max_chars_per_chunk=1500):
    """
    Build a compact prompt.
    We truncate each chunk so the model isn't overwhelmed.
    """
    legal_text_parts = []
    for i, ch in enumerate(retrieved_chunks):
        txt = ch["text"]
        txt = txt[:max_chars_per_chunk]  # truncate to avoid long context overload
        legal_text_parts.append(
            f"--- Chunk {i+1} (source: {ch['source']}) ---\n{txt}"
        )
    legal_text = "\n\n".join(legal_text_parts)

    prompt = dedent(f"""

    أنت خبير في قانون الضرائب المصري.
يجب أن تعتمد إجابتك فقط على النصوص التي تم استرجاعها.
ممنوع تماماً استخدام أي معلومات من خارج النصوص المسترجعة.
إذا لم تجد الإجابة كاملة في النصوص، يجب أن توضح بأن النص المسترجع لا يحتوي على إجابة مباشرة.

اكتب جميع الحقول باللغة العربية فقط.

    You are a senior Egyptian tax-law officer.
    Use ONLY the retrieved legal text below. Do NOT invent anything.

    Return ONLY valid JSON matching this schema exactly:
    {{
      "issue_type": "",
      "question_understanding": "",
      "legal_basis": [],
      "analysis": "",
      "application": "",
      "calculation": "",
      "conclusion": ""
    }}

    Rules:
    - Fill ALL fields (no empty strings).
    - legal_basis must list the relevant articles / fatwas you used.
    - If calculation is not applicable, write "Not applicable".

    USER QUESTION:
    {question}

    RETRIEVED LEGAL TEXT:
    {legal_text}

    Now return ONLY the JSON.
    """).strip()

    return prompt



# In[14]:


def generate_json_reasoning(prompt, max_new_tokens=700):
    chat = [
        {"role": "system", "content": "You are a tax-law expert. Output ONLY valid JSON."},
        {"role": "user", "content": prompt}
    ]

    # Qwen-native chat formatting
    input_ids = tokenizer.apply_chat_template(
        chat,
        tokenize=True,
        return_tensors="pt"
    ).to(llm_model.device)

    with torch.no_grad():
        out_ids = llm_model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False
        )

    text = tokenizer.decode(out_ids[0], skip_special_tokens=True)
    return text


# In[15]:


def clean_and_parse_json(raw_output):
    """
    Extract ONLY the LAST JSON object from the model output.
    This avoids schema-echo issues where the model prints two JSON blocks.
    """

    # Find ALL JSON objects in the output
    matches = re.findall(r"\{.*?\}", raw_output, re.DOTALL)

    if not matches:
        print("RAW OUTPUT:\n", raw_output)
        raise ValueError("No JSON object found.")

    # The REAL JSON is ALWAYS the last object
    json_str = matches[-1]

    # Clean
    json_str = json_str.replace("\n", " ").replace("\t", " ")
    json_str = re.sub(r",\s*}", "}", json_str)  # remove trailing commas

    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        print("\n--- JSON PARSE FAILED ---")
        print(json_str)
        raise




# In[16]:


def reasoning_step_2(user_question, top_k=4):
    # 1) retrieve evidence from Step 1
    retrieved = retrieve(user_question, top_k=top_k)

    # 2) build compact prompt
    prompt = build_reasoning_prompt(user_question, retrieved)

    # 3) generate raw JSON text
    raw = generate_json_reasoning(prompt)

    # 4) parse + validate
    structured = clean_and_parse_json(raw)

    return structured


# In[19]:


TAX_LETTER_TEMPLATE = """
مصلحة الضرائب المصرية
قطاع البحوث والاتفاقيات الدولية
الإدارة المركزية للبحوث والدراسات الضريبية
الإدارة العامة لبحوث ضرائب الدخل

السيد / .................................................
تحية طيبة وبعد،،،

بالإشارة إلى الاستفسار الوارد من سيادتكم بشأن:
«{issue_type}»

نفيد سيادتكم بالآتي:

أولاً: فهم السؤال:
{question_understanding}

ثانياً: الأساس القانوني:
{legal_basis}

ثالثاً: التحليل:
{analysis}

رابعاً: التطبيق:
{application}

خامساً: الحسابات:
{calculation}

ختاماً:
{conclusion}

وتفضلوا بقبول فائق الاحترام والتقدير ،،،،
تحريراً في: {date}
"""


# In[20]:


from datetime import datetime

def fill_tax_letter_template(json_data, template=TAX_LETTER_TEMPLATE):
    # Extract fields
    issue = json_data.get("issue_type", "غير محدد")
    question = json_data.get("question_understanding", "غير متوفر")
    legal_basis_list = json_data.get("legal_basis", [])
    analysis = json_data.get("analysis", "غير متوفر")
    application = json_data.get("application", "غير متوفر")
    calculation = json_data.get("calculation", "غير متوفر")
    conclusion = json_data.get("conclusion", "غير متوفر")

    # Format legal basis as bullet points
    if legal_basis_list:
        legal_basis = "\n".join([f"- {item}" for item in legal_basis_list])
    else:
        legal_basis = "لا يوجد أساس قانوني مسترجع."

    # Insert into template
    formatted_letter = template.format(
        issue_type=issue,
        question_understanding=question,
        legal_basis=legal_basis,
        analysis=analysis,
        application=application,
        calculation=calculation,
        conclusion=conclusion,
        date=datetime.now().strftime("%Y-%m-%d")
    )

    return formatted_letter


# In[21]:


def generate_final_tax_reply(user_question, top_k=5):
    # Step 2 → JSON reasoning
    reasoning_json = reasoning_step_2(user_question, top_k=top_k)

    # Step 3 → template generation
    final_letter = fill_tax_letter_template(reasoning_json)

    return final_letter


# In[26]:


from datetime import datetime

TAX_AUTHORITY_TEMPLATE = """
مصلحة الضرائب المصرية
قطاع البحوث والاتفاقيات الدولية
الادارة المركزية للبحوث والدراسات الضريبية
الادارة العامة لبحوث ضرائب الدخل

السادة / ..................................................
تحية طيبة وبعد،،،،،

ايماءً للاستفسار الوارد من سيادتكم للادارة برقم ({ref_number}) بتاريخ ({ref_date}) بشأن:
{question_understanding}

نتشرف بإفادة سيادتكم أنه استقر رأي الادارة إلى أنه:

أولاً: الأساس القانوني والتحليل:
{legal_analysis}

ثانياً: التطبيق:
{application}

ثالثاً: الحسابات:
{calculation}

رابعاً: الرأي:
{conclusion}

وتفضلوا بقبول وافر الاحترام والتقدير ،،،،،،،،

تحريراً في {today_date}
"""

def generate_official_tax_letter(json_data, ref_number="—", ref_date="—"):
    # Format legal basis
    legal_basis_list = json_data.get("legal_basis", [])
    if legal_basis_list:
        legal_basis = "\n".join([f"- {item}" for item in legal_basis_list])
    else:
        legal_basis = "- لا يوجد أساس قانوني مسترجع."

    legal_analysis = legal_basis + "\n\n" + json_data.get("analysis", "—")

    # Fill final template
    return TAX_AUTHORITY_TEMPLATE.format(
        ref_number=ref_number,
        ref_date=ref_date,
        question_understanding=json_data.get("question_understanding", "—"),
        legal_analysis=legal_analysis,
        application=json_data.get("application", "—"),
        calculation=json_data.get("calculation", "—"),
        conclusion=json_data.get("conclusion", "—"),
        today_date=datetime.now().strftime("%Y/%m/%d")
    )




def full_pipeline(user_question, ref_number="—", ref_date="—", top_k=5):
    reasoning_json = reasoning_step_2(user_question, top_k=top_k)
    letter = generate_official_tax_letter(reasoning_json, ref_number, ref_date)
    return letter





# In[25]:


print(full_pipeline(
    "ما هو المقصود بنسبة (0.0025) المقررة بالمادة 40 من قانون التأمين الصحي الشامل؟",
    ref_number="46",
    ref_date="13/1/2019"
))


# In[ ]:

def generate_final_tax_reply(user_question: str, top_k: int = 5) -> str:
    """
    Main function: takes user_question in Arabic,
    returns full Arabic tax letter as text.
    """
    # These two functions must already exist in your notebook code:
    # - reasoning_step_2
    # - fill_tax_letter_template

    reasoning_json = reasoning_step_2(user_question, top_k=top_k)
    final_letter = fill_tax_letter_template(reasoning_json)
    return final_letter



