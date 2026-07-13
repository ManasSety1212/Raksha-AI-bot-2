import os
import requests
import time

class OpenRouterService:
    def __init__(self):
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY is not set in environment variables.")
        print("[OpenRouter Service] Initialized successfully.")

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

        if not self.api_key:
            return {
                "success": False,
                "error": "OpenRouter API key is missing."
            }

        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        full_system = system_msg
        if context:
            full_system += f"\nContext: {context}"

        payload = {
            "model": "google/gemini-2.5-flash",  
            "messages": [
                {"role": "system", "content": full_system},
                {"role": "user", "content": query}
            ]
        }
        
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=60)
            inference_time = round(time.time() - start_time, 2)
            
            if response.status_code == 200:
                data = response.json()
                reply = data['choices'][0]['message']['content']
                print(f"[OpenRouter Service] Success. Inference time: {inference_time}s")
                return {
                    "success": True,
                    "provider": "openrouter",
                    "model": "google/gemini-2.5-flash",
                    "reply": reply,
                    "inference_time": inference_time
                }
            else:
                error_text = response.text
                print(f"[OpenRouter Service] Error {response.status_code}: {error_text}")
                return {
                    "success": False,
                    "error": f"OpenRouter API error: {response.status_code}",
                    "details": error_text
                }
        except Exception as e:
            print(f"[OpenRouter Service] Exception: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
