import os
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
import pickle
import google.generativeai as genai
import numpy as np
from PIL import Image
import io
import traceback
import json 
import uvicorn # 👈 Needed for Render deployment

app = FastAPI()

# --- CORS SETUP ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------
# ⚠️ API KEY SETUP
# ---------------------------------------------------------
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    print("❌ Error: Google API Key not found!")

genai.configure(api_key=GOOGLE_API_KEY, transport='rest')

# ✅ MODEL CONFIG (Smart Fallback)
EMBEDDING_MODEL = "models/text-embedding-004"
REQUESTED_MODEL = "gemini-2.5-flash" 
FALLBACK_MODEL = "gemini-1.5-flash"

# Load Brain (Criminal Law Database)
loaded_sections = []
try:
    with open("legal_brain.pkl", "rb") as f:
        loaded_sections = pickle.load(f)
    print(f"✅ Brain Loaded: {len(loaded_sections)} sections available.")
except FileNotFoundError:
    print("⚠️ Brain file not found! Mode: General Knowledge.")

# Helper Functions
def search_engine(query_text):
    if not loaded_sections: return []
    try:
        vec = genai.embed_content(model=EMBEDDING_MODEL, content=query_text)['embedding']
        results = []
        for s in loaded_sections:
            if 'embedding' in s:
                score = np.dot(vec, s['embedding'])
                if score > 0.45:
                    results.append((score, s))
        return sorted(results, key=lambda x: x[0], reverse=True)[:5]
    except Exception as e:
        print(f"Search Error: {e}")
        return []

def analyze_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes))
    # Try 2.5, fallback to 1.5 if it fails
    try:
        model = genai.GenerativeModel(REQUESTED_MODEL)
        response = model.generate_content(["Describe this image in detail for a legal context (accident, contract, injury).", image])
        return response.text
    except:
        model = genai.GenerativeModel(FALLBACK_MODEL)
        response = model.generate_content(["Describe this image in detail for a legal context.", image])
        return response.text

def get_legal_advice(story, laws):
    if laws:
        law_txt = "\n".join([f"DATABASE MATCH - SEC {l['section_id']}: {l['text']}" for l in laws])
    else:
        law_txt = "No specific match in local criminal database. USE GENERAL INDIAN LEGAL KNOWLEDGE."

    # 🌟 ULTIMATE HYBRID PROMPT
    # Contains ALL your old instructions, but formatted for JSON safety.
    prompt = f"""
    You are an Expert Indian Legal Consultant covering ALL domains:
    1. Criminal Law (BNS, BNSS, BSA)
    2. Civil Law (CPC, Contract Act, Property Act)
    3. Family Law (Hindu Marriage Act, Muslim Law, Special Marriage Act)
    4. Corporate & Consumer Law & Constitutional Law

    CONTEXT & CONVERSATION HISTORY:
    {story}
    
    RELEVANT DATABASE LAWS (If applicable):
    {law_txt}

    --------------------------------------------------------
    INSTRUCTIONS:
    
    1. **LANGUAGE (CRITICAL):** Detect the language of the 'USER SITUATION'. You MUST reply in the **SAME LANGUAGE** (Hindi, Tamil, English, Hinglish, etc.).
    
    2. **MEMORY & CONTEXT:** Analyze the LATEST user question in the context of the history.
    
    3. **UNIVERSAL KNOWLEDGE:** If no database match is found, use general Indian Law knowledge. Do not fail.

    --------------------------------------------------------
    OUTPUT FORMAT:
    Return a SINGLE VALID JSON OBJECT with keys "public" and "student".
    
    KEY 1: "public"
    (Content for Normal Citizens. Simple & Clean. NO Jargon.)
    * Structure the text using Markdown:
      ### 🛑 **DIRECT ANSWER**
      (Clear answer in user's language)
      
      ### 🏛️ **RELEVANT ACTS**
      * **Act Name:** [Section Number]
      
      ### 🚀 **NEXT STEPS**
      (3 clear steps in user's language)

    KEY 2: "student"
    (Content for Lawyers & Students. Deep Dive.)
    * Structure the text using Markdown:
      ### 🎓 **LEGAL DEEP DIVE**
      
      **Analysis:**
      (Apply specific Acts - e.g. BNS, HMA).
      
      **Precedent / Doctrine:**
      (Cite relevant legal principles/case laws).
      
      **Accuracy:** (Include 'Accuracy: 90%' score based on your confidence).
    """
    
    # 🛡️ SMART MODEL SWITCHING & JSON ENFORCEMENT
    try:
        model = genai.GenerativeModel(REQUESTED_MODEL)
        response = model.generate_content(prompt, generation_config={"response_mime_type": "application/json"})
    except Exception as e:
        print(f"⚠️ Model {REQUESTED_MODEL} failed, switching to {FALLBACK_MODEL}")
        model = genai.GenerativeModel(FALLBACK_MODEL)
        response = model.generate_content(prompt, generation_config={"response_mime_type": "application/json"})
    
    return response.text

@app.post("/ask_lawyer")
async def ask_lawyer(story: str = Form(...), file: Optional[UploadFile] = File(None)):
    try:
        print(f"📝 Received Story: {story}")
        full_context = story
        
        if file:
            print("📸 Processing Image...")
            contents = await file.read()
            image_desc = analyze_image(contents)
            full_context = f"User Story: {story}. \nImage Analysis: {image_desc}"

        # Search
        print("🔍 Searching Database...")
        matches = search_engine(full_context)
        relevant_laws = [m[1] for m in matches] if matches else []
        
        # Advice
        print("⚖️ Drafting Advice...")
        advice_json = get_legal_advice(full_context, relevant_laws)
        
        # Verify JSON validity
        try:
            parsed = json.loads(advice_json)
        except:
            # Fallback if AI messes up JSON format
            parsed = {"public": advice_json, "student": "Formatting Error"}

        return {"analysis": parsed}

    except Exception as e:
        traceback.print_exc()
        error_msg = f"### ⚠️ System Error\nI encountered a technical issue:\n\n`{str(e)}`"
        return {"analysis": {"public": error_msg, "student": "Check Logs"}}

# 🛑 CRITICAL FIX FOR RENDER DEPLOYMENT
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=10000)
