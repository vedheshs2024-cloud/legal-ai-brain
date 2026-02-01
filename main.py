from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
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
GOOGLE_API_KEY = "AIzaSyChx9Z0zvDIuv2AoYh9xiR_aC_TzZSYYpo" 
genai.configure(api_key=GOOGLE_API_KEY, transport='rest')

# ✅ MODEL: Using the latest Flash model
EMBEDDING_MODEL = "models/text-embedding-004"
MODEL_NAME = "gemini-2.5-flash" 

# Load Brain (Criminal Law Database)
loaded_sections = []
try:
    with open("legal_brain.pkl", "rb") as f:
        loaded_sections = pickle.load(f)
    print("✅ Brain Loaded Successfully! (RAG Enabled)")
except FileNotFoundError:
    print("⚠️ Brain file not found! Switching to 'General Knowledge Only' mode.")

# Helper Functions
def search_engine(query_text):
    if not loaded_sections:
        return [] 
        
    try:
        vec = genai.embed_content(model=EMBEDDING_MODEL, content=query_text)['embedding']
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

    # 🌟 THE ULTIMATE PROMPT (Universal + Memory + Multilingual)
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
    
    2. **MEMORY & CONTEXT:** Analyze the LATEST user question in the context of the history provided above. If the user asks a follow-up (e.g., "What is the punishment?"), refer to the previous topic.
    
    3. **UNIVERSAL KNOWLEDGE:** If the user's query is about Civil/Family/Corporate law and no database match is found, DO NOT fail. Use your general knowledge of Indian Law to answer accurately.

    PART 1: PUBLIC RESPONSE (Simple & Clean)
    - Audience: Normal citizens.
    - 🛑 NO technical jargon or accuracy scores here.
    - Structure (Translate headers to user's language):
      ### 🛑 **DIRECT ANSWER**
      (Clear answer in user's language)
      
      ### 🏛️ **RELEVANT ACTS**
      * **Act Name:** [Section Number]
      
      ### 🚀 **NEXT STEPS**
      (3 clear steps in user's language)

    PART 2: SEPARATOR
    - Write exactly "|||"

    PART 3: STUDENT RESPONSE (Deep Dive)
    - Audience: Lawyers & Students.
    - ✅ INCLUDE Accuracy Scores if you are confident: `Accuracy: 90%`.
    - Cite Case Laws (AIR, SCC) if applicable.
    - Structure:
      ### 🎓 **LEGAL DEEP DIVE**
      
      **Analysis:**
      (Apply the specific Act—e.g., Section 13B of HMA for Divorce, or Section 303 BNS for Theft).
      
      **Precedent / Doctrine:**
      (Cite relevant legal principles).
    --------------------------------------------------------
    """
    
    model = genai.GenerativeModel(MODEL_NAME)
    response = model.generate_content(prompt).text.strip()
    
    if "|||" in response:
        parts = response.split("|||")
        public_part = parts[0].strip()
        student_part = parts[1].strip()
    else:
        public_part = response
        student_part = "### 🎓 **No Additional Technical Detail**\n(Refer to the public answer)."

    return json.dumps({
        "public": public_part,
        "student": student_part
    })

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

        # Search (Hybrid Strategy)
        print("🔍 Searching Database...")
        matches = search_engine(full_context)
        
        relevant_laws = [m[1] for m in matches] if matches else []
        
        if not relevant_laws:
            print("⚠️ No database match found. Switching to General Knowledge Mode.")

        # Advice
        print("⚖️ Drafting Advice...")
        advice = get_legal_advice(full_context, relevant_laws)
        
        return {"analysis": advice}

    except Exception as e:
        print("❌ CRITICAL ERROR:")
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)})