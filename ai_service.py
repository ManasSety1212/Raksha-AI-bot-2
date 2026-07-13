import os
import traceback

# Import services with fallback path support
GeminiService = None
HuggingFaceService = None
OpenRouterService = None

try:
    from gemini_service import GeminiService
    from huggingface_service import HuggingFaceService
    from openrouter_service import OpenRouterService
except ImportError:
    try:
        from services.gemini_service import GeminiService
        from services.huggingface_service import HuggingFaceService
        from services.openrouter_service import OpenRouterService
    except ImportError:
        try:
            from backend.services.gemini_service import GeminiService
            from backend.services.huggingface_service import HuggingFaceService
            from backend.services.openrouter_service import OpenRouterService
        except ImportError:
            try:
                from backEnd.services.gemini_service import GeminiService
                from backEnd.services.huggingface_service import HuggingFaceService
                from backEnd.services.openrouter_service import OpenRouterService
            except ImportError as err:
                print(f"[AI Service Class] Critical imports failed: {err}")

class AIService:
    def __init__(self):
        self.engine = None
        self.provider_name = "disconnected"
        self.model_name = "none"
        self.init_errors = []

        # 1. Try Google Gemini Service as primary and default
        try:
            print("[AI Service Class] Initializing Google Gemini Service...")
            if GeminiService is None:
                raise ValueError("GeminiService class import is None/missing.")
            
            self.engine = GeminiService()
            self.provider_name = "gemini"
            self.model_name = self.engine.model_name
            print("✓ Gemini initialized successfully")
            return
        except Exception as e:
            tb = traceback.format_exc()
            self.init_errors.append(f"Gemini initialization failed: {str(e)}\n{tb}")
            print(f"✗ Gemini initialization failed:\n{e}")
            traceback.print_exc()

        # 2. Try HuggingFace (Gemini fallback wrapper) ONLY if Gemini initialization fails
        try:
            print("[AI Service Class] Falling back to HuggingFace (Gemini Adapter) Service...")
            if HuggingFaceService is None:
                raise ValueError("HuggingFaceService class import is None/missing.")
            
            self.engine = HuggingFaceService()
            self.provider_name = "huggingface"
            self.model_name = "gemini-1.5-flash"
            print("✓ HuggingFace Service initialized successfully as fallback.")
            return
        except Exception as e:
            tb = traceback.format_exc()
            self.init_errors.append(f"HuggingFace initialization failed: {str(e)}\n{tb}")
            print(f"[AI Service Class] HuggingFace fallback initialization failed: {e}")
            traceback.print_exc()

        # 3. Try OpenRouter Service as ultimate fallback
        try:
            print("[AI Service Class] Falling back to OpenRouter Service...")
            if OpenRouterService is None:
                raise ValueError("OpenRouterService class import is None/missing.")
                
            self.engine = OpenRouterService()
            self.provider_name = "openrouter"
            self.model_name = "google/gemini-2.5-flash"
            print("✓ OpenRouter Service initialized successfully as ultimate fallback.")
            return
        except Exception as e:
            tb = traceback.format_exc()
            self.init_errors.append(f"OpenRouter initialization failed: {str(e)}\n{tb}")
            print(f"[AI Service Class] OpenRouter fallback initialization failed: {e}")
            traceback.print_exc()

        # 4. If all fail
        self.engine = None
        self.provider_name = "disconnected"
        self.model_name = "none"
        combined_logs = "\n\n".join(self.init_errors)
        print("="*60)
        print("CRITICAL LOG: ALL AI SERVICES FAILED TO INITIALIZE!")
        print(combined_logs)
        print("="*60)

    def get_chat_response(self, query, section, context=None):
        if not self.engine:
            combined_errors = "; ".join([e.split("\n")[0] for e in self.init_errors])
            return {
                "success": False,
                "error": f"Bot engine not ready. Startup errors: {combined_errors}"
            }
        return self.engine.get_chat_response(query, section, context)

    def generate_study_plan(self, exam_data):
        if not self.engine:
            return "Error: Bot engine not ready."
        if hasattr(self.engine, 'generate_study_plan'):
            return self.engine.generate_study_plan(exam_data)
        else:
            prompt = (
                f"Generate a full study plan for the exam: {exam_data.get('examName')}. "
                f"Exam Date: {exam_data.get('examDate')}. Syllabus: {exam_data.get('syllabus')}. "
                "The plan MUST include: Exam overview, Syllabus breakdown, Roadmap till exam date, "
                "Daily targets, Weekly targets, Revision plan, Mock test plan, Resources, and a Final 7-day strategy. "
                "Format the response clearly using Markdown."
            )
            result = self.get_chat_response(prompt, "education")
            return result.get("reply") if result.get("success") else "Error generating study plan."
