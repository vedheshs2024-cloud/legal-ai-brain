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
# ⚠️ YOUR API KEY
# ---------------------------------------------------------
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    print("❌ Error: Google API Key not found in Environment Variables!")

genai.configure(api_key=GOOGLE_API_KEY, transport='rest')

# ✅ MODEL: Keeping your requested model name
EMBEDDING_MODEL = "models/text-embedding-004"
MODEL_NAME = "gemini-2.5-flash" 

# Load Brain (Criminal Law Database)
loaded_sections = []
try:
    with open("legal_brain.pkl", "rb") as f:
        loaded_sections = pickle.load(f)
    print(f"✅ Brain Loaded Successfully! ({len(loaded_sections)} sections available)")
except FileNotFoundError:
    print("⚠️ Brain file not found! Switching to 'General Knowledge Only' mode.")

# Helper Functions
def search_engine(query_text):
    if not loaded_sections:
        return [] 
        
    try:
        # 1. Get Query Embedding
        vec = genai.embed_content(model=EMBEDDING_MODEL, content=query_text)['embedding']
        
        # 2. Vectorized Search (Faster & Cleaner)
        results = []
        for s in loaded_sections:
            if 'embedding' in s:
                score = np.dot(vec, s['embedding'])
                if score > 0.45: # Threshold
                    results.append((score, s))
        
        return sorted(results, key=lambda x: x[0], reverse=True)[:5]
    except Exception as e:
        print(f"Search Error: {e}")
        return []

def analyze_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes))
    model = genai.GenerativeModel(MODEL_NAME) 
    prompt = "Describe this image in detail for a legal context (e.g., accident, contract, injury, property dispute)."
    response = model.generate_content([prompt, image])
    return response.text

def get_legal_advice(story, laws):
    # Logic: Use Database matches if found, otherwise fallback to General Knowledge.
    if laws:
        law_txt = "\n".join([f"DATABASE MATCH - SEC {l['section_id']}: {l['text']}" for l in laws])
    else:
        law_txt = "No specific match in local criminal database. USE GENERAL INDIAN LEGAL KNOWLEDGE (Civil, Family, Corporate, Constitutional, etc.)."

    # 🌟 THE ULTIMATE PROMPT (Restored your specific instructions + JSON Mode)
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
    
    2. **MEMORY & CONTEXT:** Analyze the LATEST user question in the context of the history provided above. If the user asks a follow-up, refer to the previous topic.
    
    3. **UNIVERSAL KNOWLEDGE:** If the user's query is about Civil/Family/Corporate law and no database match is found, DO NOT fail. Use your general knowledge.

    OUTPUT FORMAT:
    You must return a SINGLE valid JSON object with keys "public" and "student". 
    Do not use markdown code blocks (like ```json). Just the raw JSON.

    KEY 1: "public" (Simple & Clean)
    - Audience: Normal citizens.
    - Structure (Translate headers to user's language):
      ### 🛑 **DIRECT ANSWER**
      (Clear answer in user's language)
      
      ### 🏛️ **RELEVANT ACTS**
      * **Act Name:** [Section Number]
      
      ### 🚀 **NEXT STEPS**
      (3 clear steps in user's language)

    KEY 2: "student" (Deep Dive)
    - Audience: Lawyers & Students.
    - Structure:
      ### 🎓 **LEGAL DEEP DIVE**
      **Analysis:** (Apply specific Acts).
      **Precedent / Doctrine:** (Cite case laws/principles).
      **Accuracy:** (Include 'Accuracy: 90%' score).
    --------------------------------------------------------
    """
    
    model = genai.GenerativeModel(MODEL_NAME)
    
    # ✅ FORCE JSON MODE (Fixes the splitting issues while keeping your prompt)
    response = model.generate_content(
        prompt,
        generation_config={"response_mime_type": "application/json"}
    )
    
    return response.text

@app.post("/ask_lawyer")
async def ask_lawyer(
    story: str = Form(...), 
    file: Optional[UploadFile] = File(None)
):
    try:
        print(f"📝 Received Story: {story}")
        full_context = story
        
        # Image Handling
        if file:
            print("📸 Processing Image...")
            contents = await file.read()
            image_desc = analyze_image(contents)
            print(f"   Image Analysis: {image_desc}")
            full_context = f"User says: {story}. \nPhoto shows: {image_desc}"

        # Search
        print("🔍 Searching Database...")
        matches = search_engine(full_context)
        
        relevant_laws = [m[1] for m in matches] if matches else []
        
        if not relevant_laws:
            print("⚠️ No database match found. Switching to General Knowledge Mode.")

        # Advice
        print("⚖️ Drafting Advice...")
        advice_json_string = get_legal_advice(full_context, relevant_laws)
        
        # ✅ VERIFY JSON (Prevents frontend crashes)
        try:
            parsed_advice = json.loads(advice_json_string)
        except json.JSONDecodeError:
            # Fallback if AI makes a tiny syntax error
            parsed_advice = {
                "public": advice_json_string, 
                "student": "### 🎓 Error\nCould not format deep dive."
            }
        
        # Return Object (Not String)
        return {"analysis": parsed_advice}

    except Exception as e:
        print("❌ CRITICAL ERROR:")
        traceback.print_exc()
        error_message = f"### ⚠️ System Error\nI encountered a technical issue:\n\n`{str(e)}`"
        # Return valid JSON structure even on error
        return {"analysis": {"public": error_message, "student": "Check server logs for traceback."}}
