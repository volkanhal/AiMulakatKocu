from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import google.generativeai as genai
import json

# --- AYARLAR ---
# Senin API Anahtarın
API_KEY = "AIzaSyCbAh8XhQLjjjtc3vZZiPx8Zzwz5fjI_UQ"

# Gemini Ayarları
genai.configure(api_key=API_KEY)

# DİKKAT: Senin listendeki en hızlı modeli seçtik!
model = genai.GenerativeModel('gemini-2.5-flash')

app = FastAPI()

# --- DATA MODELLERİ ---
class CVAnalysisRequest(BaseModel):
    cv_text: str
    job_description: str

class InterviewContext(BaseModel):
    history: List[dict] 
    cv_text: str
    job_description: str

# --- YARDIMCI FONKSİYON ---
def clean_json_string(json_string):
    # AI bazen ```json etiketiyle cevap verir, temizliyoruz
    return json_string.replace("```json", "").replace("```", "").strip()

# --- 1. ENDPOINT: CV ANALİZİ ---
@app.post("/analyze-cv")
async def analyze_cv(request: CVAnalysisRequest):
    print("----- CV ANALİZİ İSTEĞİ GELDİ -----")
    print(f"Model Kullanılıyor: {model.model_name}")
    
    prompt = f"""
    Sen teknik bir mülakatçısın.
    İŞ İLANI: {request.job_description}
    ADAY CV: {request.cv_text}
    
    GÖREVİN:
    Bu adayın CV'sindeki 3 tane zayıf noktayı tespit et.
    Sadece ve sadece aşağıdaki JSON formatında cevap ver:
    {{
        "weak_points": [
            "1. nokta",
            "2. nokta",
            "3. nokta"
        ]
    }}
    """

    try:
        response = model.generate_content(prompt)
        print("AI Cevabı:", response.text) # Terminalde cevabı görelim
        cleaned_response = clean_json_string(response.text)
        return json.loads(cleaned_response)
    except Exception as e:
        print("HATA:", e)
        return {"error": "AI Cevap veremedi", "detay": str(e)}

# --- 2. ENDPOINT: SORU SORMA ---
@app.post("/next-question")
async def next_question(context: InterviewContext):
    # Gemini Sohbet Geçmişi Ayarı
    chat_history = []
    for msg in context.history:
        role = "model" if msg["role"] == "assistant" else "user"
        chat_history.append({"role": role, "parts": [msg["content"]]})

    chat = model.start_chat(history=chat_history)
    
    system_instruction = f"""
    Sen bir işe alım uzmanısın.
    ADAY CV: {context.cv_text}
    POZİSYON: {context.job_description}
    
    Görevin: Adayın son cevabını analiz et ve mantıklı tek bir soru sor.
    Sadece JSON formatında cevap ver:
    {{
        "message": "Soru metni",
        "intent": "question"
    }}
    """
    
    try:
        response = chat.send_message(system_instruction)
        cleaned_response = clean_json_string(response.text)
        return json.loads(cleaned_response)
    except Exception as e:
        return {"error": "Soru üretilemedi", "detay": str(e)}
    # --- 3. ENDPOINT: MÜLAKAT RAPORU (YARGIÇ MODU) ---
@app.post("/generate-report")
async def generate_report(context: InterviewContext):
    # Gemini Sohbet Geçmişi Formatı
    chat_history_text = ""
    for msg in context.history:
        role = "AI" if msg["role"] == "assistant" else "ADAY"
        chat_history_text += f"{role}: {msg['content']}\n"

    system_instruction = f"""
    GÖREV: Aşağıdaki mülakat geçmişini analiz et ve bir karne oluştur.
    
    POZİSYON: {context.job_description}
    MÜLAKAT GEÇMİŞİ:
    {chat_history_text}
    
    İSTENEN ÇIKTI (Sadece JSON):
    {{
        "total_score": 0-100 arası puan,
        "pros": ["Adayın iyi olduğu yön 1", "Adayın iyi olduğu yön 2"],
        "cons": ["Adayın eksik olduğu yön 1", "Adayın eksik olduğu yön 2"],
        "question_analysis": [
            {{
                "question": "AI'ın sorduğu soru",
                "candidate_answer": "Adayın cevabı",
                "score": 1-10 arası puan,
                "feedback": "Cevap yeterli miydi? Nasıl daha iyi olabilirdi?"
            }}
        ],
        "final_decision": "Olumlu/Olumsuz/Kararsız"
    }}
    """
    
    try:
        response = model.generate_content(system_instruction)
        cleaned_response = clean_json_string(response.text)
        return json.loads(cleaned_response)
    except Exception as e:
        return {"error": "Rapor oluşturulamadı", "detay": str(e)}