import requests
import os
import time

class HuggingFaceService:
    def __init__(self):
        self.gemini_api_key = os.getenv("GEMINI_API_KEY")
        self.fallback_key = "AIzaSyBzfBhpdOw7bwt_PMykOP0icdGK7wkcaM4"
        
        if not self.gemini_api_key or not self.gemini_api_key.startswith("AIzaSy"):
            print("[Gemini Service] Primary key invalid/missing. Falling back to default SDK key.")
            self.gemini_api_key = self.fallback_key
        else:
            print("[Gemini Service] Initialized successfully with primary API key. Model: gemini-1.5-flash")

    def get_system_instruction(self, section):
        base = (
            "You are Raksha AI Bot, a safety, education, and tech assistant inside the Raksha AI women safety app. "
            "Keep answers practical, simple, India-focused, and helpful. "
            "For legal help, provide general guidance only and advise contacting police/lawyer for urgent cases. "
            "For exams, never invent dates. If data is missing, say it is not clearly mentioned."
        )
        
        section_rules = {
            "safety": (
                "SECTION: SAFETY. You must only answer about: women safety, SOS, emergency steps, police help, "
                "complaint filing, Raksha AI app usage, and cyber safety. "
                "If the user asks an unrelated question, politely say: 'This section only supports safety related help. Please switch section.'"
            ),
            "education": (
                "SECTION: EDUCATION. You must only answer about: competitive exams, government exams, study plans, "
                "syllabus, preparation strategy, latest live forms, and exam doubts. "
                "If the user asks an unrelated question, politely say: 'This section only supports education and exam help. Please switch section.'"
            ),
            "tech": (
                "SECTION: TECH. You must only answer about: app usage, phone safety, cyber security, technical doubts, "
                "basic coding help, and Raksha AI technical features. "
                "If the user asks an unrelated question, politely say: 'This section only supports technical help. Please switch section.'"
            ),
            "legal": (
                "SECTION: LEGAL HELP. You are guiding the user on their rights. "
                "If mode is 'Police': Give step-by-step emergency action, FIR/complaint guidance, helpline suggestions, and nearest police station guidance. "
                "If mode is 'Lawyer': Explain relevant sections, rights, documentation, and useful proof. "
                "ADD DISCLAIMER: 'This is general legal guidance. For serious cases, contact police or a qualified lawyer.'"
            )
        }
        
        return f"{base} {section_rules.get(section, '')}"

    def get_chat_response(self, query, section, context=None):
        start_time = time.time()
        system_msg = self.get_system_instruction(section)

        if not self.gemini_api_key:
            return {
                "success": False,
                "error": "Gemini API is not configured (GEMINI_API_KEY missing)."
            }

        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={self.gemini_api_key}"
        headers = {"Content-Type": "application/json"}
        
        full_system = system_msg
        if context:
            full_system += f"\nContext: {context}"
            
        payload = {
            "contents": [
                {
                    "parts": [
                        {"text": query}
                    ]
                }
            ],
            "systemInstruction": {
                "parts": [
                    {"text": full_system}
                ]
            }
        }
        
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=60)
            inference_time = round(time.time() - start_time, 2)
            
            if response.status_code == 200:
                data = response.json()
                reply = data['candidates'][0]['content']['parts'][0]['text']
                print(f"[Gemini Service] Success. Inference time: {inference_time}s")
                return {
                    "success": True,
                    "provider": "gemini",
                    "model": "gemini-1.5-flash",
                    "reply": reply,
                    "inference_time": inference_time
                }
            else:
                error_text = response.text
                print(f"[Gemini Service] Error {response.status_code}: {error_text}")
                return {
                    "success": False,
                    "error": f"Gemini API error: {response.status_code}",
                    "details": error_text
                }
        except Exception as e:
            print(f"[Gemini Service] Exception: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
            
    def generate_study_plan(self, exam_data):
        prompt = (
            f"Generate a full study plan for the exam: {exam_data.get('examName')}. "
            f"Exam Date: {exam_data.get('examDate')}. Syllabus: {exam_data.get('syllabus')}. "
            "The plan MUST include: Exam overview, Syllabus breakdown, Roadmap till exam date, "
            "Daily targets, Weekly targets, Revision plan, Mock test plan, Resources, and a Final 7-day strategy. "
            "Format the response clearly using Markdown."
        )
        result = self.get_chat_response(prompt, "education")
        return result.get("reply") if result.get("success") else "Error generating study plan."
