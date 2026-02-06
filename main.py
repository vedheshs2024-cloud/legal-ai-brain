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
import uvicorn 
import re 

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 🔑 DUAL KEY SETUP (The New Feature)
# It tries to find Key 1. If Key 2 exists, it keeps it ready as a backup.
KEY_1 = os.environ.get("GOOGLE_API_KEY")
KEY_2 = os.environ.get("GOOGLE_API_KEY_2")

if not KEY_1:
    print("❌ Error: Google API Key 1 not found!")

# ✅ CONFIG
EMBEDDING_MODEL = "models/text-embedding-004"
REQUESTED_MODEL = "gemini-2.5-flash" 
# ⚠️ FIX: Changed to 'gemini-pro' to prevent 404 errors on your system
FALLBACK_MODEL = "gemini-pro" 

# Load Brain
loaded_sections = []
try:
    with open("legal_brain.pkl", "rb") as f:
        loaded_sections = pickle.load(f)
    print(f"✅ Brain Loaded: {len(loaded_sections)} sections available.")
except FileNotFoundError:
    print("⚠️ Brain file not found! Mode: General Knowledge.")

# 🛡️ NEW HELPER: Tries Key 1, then Key 2
def generate_with_backup(model_name, prompt, config=None):
    # List of available keys
    keys = [k for k in [KEY_1, KEY_2] if k]
    
    for i, api_key in enumerate(keys):
        try:
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel(model_name)
            return model.generate_content(prompt, generation_config=config)
        except Exception as e:
            print(f"⚠️ Key {i+1} Failed or Busy: {e}")
            # If this was the last key, stop trying and crash
            if i == len(keys) - 1:
                raise e
            # Otherwise, loop continues to the next key...

def search_engine(query_text):
    if not loaded_sections: return []
    try:
        # Use Key 1 for search (simple)
        if KEY_1: genai.configure(api_key=KEY_1)
        
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
    try:
        # Try Main Model with Backup Keys
        response = generate_with_backup(REQUESTED_MODEL, ["Describe this image in detail for a legal context.", image])
        return response.text
    except:
        print(f"⚠️ Switching to Fallback for Image...")
        # Try Fallback Model with Backup Keys
        response = generate_with_backup(FALLBACK_MODEL, ["Describe this image in detail for a legal context.", image])
        return response.text

def get_legal_advice(story, laws):
    if laws:
        law_txt = "\n".join([f"DATABASE MATCH - SEC {l['section_id']}: {l['text']}" for l in laws])
    else:
        law_txt = "No specific match. Using General Knowledge."

    prompt = f"""
    You are an Expert Indian Legal Consultant.
    CONTEXT: {story}
    RELEVANT LAWS: {law_txt}

    INSTRUCTIONS:
    1. **STRICT LANGUAGE RULE:** - If input is ENGLISH -> Reply in ENGLISH.
       - If input is HINDI -> Reply in HINDI.
       - If input is TAMIL -> Reply in TAMIL.
    
    2. **FORMAT:** Return ONLY a Single Valid JSON Object. Do not use Markdown blocks.

    JSON SCHEMA:
    {{
        "public": "### 🛑 **DIRECT ANSWER**\\n(Answer in user language)\\n### 🏛️ **ACTS**\\n(Acts)\\n### 🚀 **STEPS**\\n(Steps)",
        "student": "### 🎓 **DEEP DIVE**\\nAnalysis, Precedents, Accuracy %"
    }}
    """
    
    try:
        # Try Main Model with Backup Keys
        response = generate_with_backup(REQUESTED_MODEL, prompt, config={"response_mime_type": "application/json"})
        return response.text
    except:
        print(f"⚠️ Switching to Fallback for Text...")
        # Try Fallback Model with Backup Keys
        response = generate_with_backup(FALLBACK_MODEL, prompt, config={"response_mime_type": "application/json"})
        return response.text

@app.post("/ask_lawyer")
async def ask_lawyer(story: str = Form(...), file: Optional[UploadFile] = File(None)):
    try:
        full_context = story
        if file:
            contents = await file.read()
            image_desc = analyze_image(contents)
            full_context = f"User Story: {story}. \nImage Analysis: {image_desc}"

        matches = search_engine(full_context)
        relevant_laws = [m[1] for m in matches] if matches else []
        
        advice_json = get_legal_advice(full_context, relevant_laws)

        # 🧹 CLEANING STEP
        cleaned_json = advice_json.strip()
        if cleaned_json.startswith("```json"):
            cleaned_json = cleaned_json.replace("```json", "", 1)
        if cleaned_json.startswith("```"):
            cleaned_json = cleaned_json.replace("```", "", 1)
        if cleaned_json.endswith("```"):
            cleaned_json = cleaned_json.replace("```", "", 1)
        cleaned_json = cleaned_json.strip()
        
        try:
            parsed = json.loads(cleaned_json)
        except:
            parsed = {"public": cleaned_json, "student": "Formatting Error"}

        return {"analysis": parsed}

    except Exception as e:
        traceback.print_exc()
        return {"analysis": {"public": f"⚠️ Error: {str(e)}", "student": "Check Logs"}}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=10000)