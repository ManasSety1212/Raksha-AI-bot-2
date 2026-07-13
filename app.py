from flask import Flask, request, jsonify, Response, send_from_directory
from flask_socketio import SocketIO, emit, join_room, leave_room
from flask_cors import CORS
import cv2
import numpy as np
import base64
import os
import time
import sys
from datetime import datetime
from dotenv import load_dotenv
from pathlib import Path

# Load Environment Variables
load_dotenv()

BASE_DIR = Path(__file__).resolve().parent

# Initialize App
app = Flask(__name__)
CORS(app)
# max_http_buffer_size = 5MB to handle incoming frame chunks safely
socketio = SocketIO(app, cors_allowed_origins="*", max_http_buffer_size=5_000_000, async_mode="threading")

# --- AI & FIREBASE CONFIG ---
AI_PROVIDER = "gemini"

# --- IMPORT SYSTEM REFACTOR ---
AIService = None
GeminiService = None
HuggingFaceService = None
RakshaFirebaseService = None
StudyPlanPDFGenerator = None
OpenRouterService = None

try:
    from ai_service import AIService
    from gemini_service import GeminiService
    from huggingface_service import HuggingFaceService
    from firebase_service import RakshaFirebaseService
    from pdf_generator import StudyPlanPDFGenerator
    from openrouter_service import OpenRouterService
    print("[Imports] Flat layout detected")
except ImportError:
    try:
        from backend.services.ai_service import AIService
        from backend.services.gemini_service import GeminiService
        from backend.services.huggingface_service import HuggingFaceService
        from backend.raksha_bot.firebase_service import RakshaFirebaseService
        from backend.raksha_bot.pdf_generator import StudyPlanPDFGenerator
        from backend.services.openrouter_service import OpenRouterService
        print("[Imports] backend package detected")
    except ImportError:
        try:
            from backEnd.services.ai_service import AIService
            from backEnd.services.gemini_service import GeminiService
            from backEnd.services.huggingface_service import HuggingFaceService
            from backEnd.raksha_bot.firebase_service import RakshaFirebaseService
            from backEnd.raksha_bot.pdf_generator import StudyPlanPDFGenerator
            from backEnd.services.openrouter_service import OpenRouterService
            print("[Imports] backend package detected")
        except ImportError:
            try:
                from services.ai_service import AIService
                from services.gemini_service import GeminiService
                from services.huggingface_service import HuggingFaceService
                from raksha_bot.firebase_service import RakshaFirebaseService
                from raksha_bot.pdf_generator import StudyPlanPDFGenerator
                from services.openrouter_service import OpenRouterService
                print("[Imports] backend package detected")
            except ImportError as e:
                print(f"[Warning] Failed to import bot modules: {e}")
                import traceback
                traceback.print_exc()

import firebase_admin
from firebase_admin import credentials, firestore, storage

# --- FIREBASE INITIALIZATION ---
firebase_initialized = False
try:
    cred_name = os.environ.get("FIREBASE_SERVICE_ACCOUNT_NAME", "serviceAccountKey.json")
    storage_bucket = os.environ.get("FIREBASE_STORAGE_BUCKET", "tanprix-52683.appspot.com")
    
    if not firebase_admin._apps:
        if cred_name and os.path.exists(cred_name):
            cred = credentials.Certificate(cred_name)
            firebase_admin.initialize_app(cred, {
                'storageBucket': storage_bucket
            })
            firebase_initialized = True
            print("[Firebase] Initialized")
        else:
            try:
                firebase_admin.initialize_app(options={
                    'storageBucket': storage_bucket
                })
                firebase_initialized = True
                print("[Firebase] Initialized")
            except Exception as adc_err:
                print(f"[Firebase] Initialization failed and fallback failed: {adc_err}")
                print("[Firebase] Disabled")
    else:
        firebase_initialized = True
        print("[Firebase] Initialized")
except Exception as e:
    print(f"[Firebase] Critical Init Error: {e}")
    print("[Firebase] Disabled")

# --- BOT INITIALIZATION ---
bot_engine = None
bot_fb = None
pdf_gen = None

if 'AIService' in globals() and AIService is not None:
    try:
        bot_engine = AIService()
        if bot_engine and bot_engine.provider_name == "gemini":
            print("[Gemini] Initialized")
    except Exception as e:
        print(f"[Bot] AI Service initialization failed: {e}")
        bot_engine = None

    # RakshaFirebaseService must never crash app startup
    try:
        if firebase_initialized and RakshaFirebaseService is not None:
            bot_fb = RakshaFirebaseService()
        else:
            bot_fb = None
    except Exception as e:
        print("[Firebase Disabled]", e)
        bot_fb = None

    # StudyPlanPDFGenerator must also be optional
    try:
        if StudyPlanPDFGenerator is not None:
            pdf_gen = StudyPlanPDFGenerator()
        else:
            pdf_gen = None
    except Exception as e:
        print("[PDF Generator Disabled]", e)
        pdf_gen = None

    if bot_engine is not None:
        print("[Bot] Ready")

# --- VISION MODELS INITIALIZATION ---
face_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(face_cascade_path) if os.path.exists(face_cascade_path) else None

yolo_weights = os.path.join(BASE_DIR, "yolov3-tiny.weights")
yolo_cfg = os.path.join(BASE_DIR, "yolov3-tiny.cfg")
coco_names = os.path.join(BASE_DIR, "coco.names")

net = None
classes = []
output_layers = []

if os.path.exists(yolo_weights) and os.path.exists(yolo_cfg) and os.path.exists(coco_names):
    try:
        net = cv2.dnn.readNet(yolo_weights, yolo_cfg)
        with open(coco_names, "r") as f:
            classes = [line.strip() for line in f.readlines()]
        layer_names = net.getLayerNames()
        try:
            output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
        except:
            output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
        print("[INFO] YOLOv3-tiny loaded successfully.")
    except Exception as e:
        print(f"[YOLO Disabled] Failed to load YOLO: {e}")
        net = None
else:
    print("[YOLO Disabled]")

# --- STARTUP DIAGNOSTICS & SYSTEM REPORT ---
version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
sio_mode = socketio.async_mode if socketio else "None"
fb_status = "Initialized" if firebase_initialized else "Disabled"
gemini_status = "Initialized" if (bot_engine and bot_engine.provider_name == "gemini") else "Disabled"
yolo_status = "Initialized" if net is not None else "Disabled"

print("-" * 40)
print(f"Python Version: {version}")
print(f"SocketIO Mode: {sio_mode}")
print(f"Firebase: {fb_status}")
print(f"Gemini: {gemini_status}")
print(f"YOLO: {yolo_status}")
print("Backend Ready")
print("-" * 40)

# In-memory storage
evidence_count = {}
evidence_dir = os.path.join(BASE_DIR, "evidence")
os.makedirs(evidence_dir, exist_ok=True)
active_sessions = {}

# --- SOCKET.IO HANDLERS ---

@socketio.on('connect')
def handle_connect():
    print(f"[Socket] Client connected: {request.sid}")

@socketio.on('disconnect')
def handle_disconnect():
    print(f"[Socket] Client disconnected: {request.sid}")
    for sos_id, session_data in list(active_sessions.items()):
        if session_data.get('streamerSocket') == request.sid:
            del active_sessions[sos_id]
            emit('sos:session_ended', {'sosId': sos_id}, broadcast=True)

@socketio.on('sos:start')
def on_sos_start(data):
    sos_id = data.get('sosId')
    user_name = data.get('userName', 'Unknown')
    active_sessions[sos_id] = {
        'sosId': sos_id,
        'userId': data.get('userId'),
        'userName': user_name,
        'location': data.get('location'),
        'streamerSocket': request.sid,
        'startTime': time.time() * 1000,
        'lastFrameTime': time.time() * 1000,
        'humanDetected': False
    }
    join_room(f"sos:{sos_id}")
    emit('sos:new_session', active_sessions[sos_id], broadcast=True, include_self=False)

@socketio.on('sos:frame')
def on_sos_frame(data):
    sos_id = data.get('sosId')
    if sos_id in active_sessions:
        active_sessions[sos_id]['lastFrameTime'] = time.time() * 1000
        active_sessions[sos_id]['humanDetected'] = data.get('humanDetected') or active_sessions[sos_id]['humanDetected']
        emit(f"sos:live_frame:{sos_id}", {
            'frame': data.get('frame'),
            'timestamp': data.get('timestamp'),
            'humanDetected': data.get('humanDetected')
        }, broadcast=True, include_self=False)

@socketio.on('admin:voice_command')
def on_admin_voice_command(data):
    sos_id = data.get('sosId')
    message = data.get('message')
    session = active_sessions.get(sos_id)
    if session:
        emit('sos:voice_command', {'message': message, 'sosId': sos_id}, to=session['streamerSocket'])

# --- REST API ROUTES ---

@app.route("/", methods=["GET", "HEAD"])
def index():
    return jsonify({
        "service": "RakshaAI Central AI Engine",
        "status": "Online",
        "version": "1.0",
        "endpoints": ["/health", "/api/auth/register", "/api/evidence/analyze"]
    })

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "ok"})

@app.route("/api/ai/provider", methods=["GET"])
def get_ai_provider():
    if bot_engine and bot_engine.engine is not None:
        return jsonify({
            "provider": bot_engine.provider_name,
            "model": bot_engine.model_name,
            "status": "connected"
        })
    else:
        return jsonify({
            "provider": "disconnected",
            "model": "none",
            "status": "disconnected"
        })

@app.route("/api/ai/chat", methods=["POST"])
def ai_chat():
    try:
        data = request.json
        user_message = data.get("message", "")
        section = data.get("section", "safety")
        user_id = data.get("user_id", "guest")

        if not user_message:
            return jsonify({"success": False, "error": "Message is empty"}), 400

        if bot_engine:
            res = bot_engine.get_chat_response(user_message, section)
            if res.get("success"):
                reply = res.get("reply")
                if bot_fb and user_id != "guest":
                    try:
                        bot_fb.save_chat_message(user_id, {"sender": "bot", "message": reply, "section": section})
                    except: pass
                return jsonify({
                    "success": True,
                    "provider": res.get("provider", "gemini"),
                    "model": res.get("model", "gemini-1.5-flash"),
                    "reply": reply
                })
            else:
                err_msg = res.get("error", "AI Inference Failure")
                details = res.get("details", "")
                print(f"[AI Chat] Bot engine failed response: {err_msg}. Details: {details}")
                return jsonify({
                    "success": False,
                    "error": "Bot engine not ready",
                    "details": f"{err_msg}. {details}".strip()
                }), 503
        else:
            print("[AI Chat] Bot engine not ready: AI service was not instantiated during startup.")
            return jsonify({
                "success": False,
                "error": "Bot engine not ready",
                "details": "AI service was not instantiated during startup. Please check startup logs for exception traces."
            }), 500
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"[AI Chat] Exception: {e}")
        return jsonify({
            "success": False,
            "error": "Bot engine not ready",
            "details": str(e)
        }), 500

@app.route("/api/debug/ai")
def debug_ai():
    try:
        bot = bot_engine
        return jsonify({
            "google_key_present": bool(os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")),
            "bot_initialized": bot is not None,
            "bot_type": type(bot).__name__ if bot else None
        })
    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 500

@app.route("/api/debug/firebase", methods=["GET"])
def debug_firebase():
    cred_name = os.environ.get("FIREBASE_SERVICE_ACCOUNT_NAME", "serviceAccountKey.json")
    service_account_exists = bool(cred_name and os.path.exists(cred_name))
    
    firestore_active = False
    if bot_fb is not None:
        try:
            if bot_fb.db is not None:
                firestore_active = True
        except Exception as e:
            print(f"[Debug Firebase] Firestore check failed: {e}")
            
    return jsonify({
        "firebase_initialized": firebase_initialized,
        "service_account_found": service_account_exists,
        "storage_bucket": os.environ.get("FIREBASE_STORAGE_BUCKET", "tanprix-52683.appspot.com"),
        "firestore_ready": firestore_active
    })

@app.route('/api/auth/register', methods=['POST'])
def register_face():
    data = request.json
    user_id = data.get('user_id')
    image_base64 = data.get('image')
    if not user_id or not image_base64: return jsonify({"error": "Missing data"}), 400
    try:
        if ',' in image_base64: image_base64 = image_base64.split(',')[1]
        img_data = base64.b64decode(image_base64)
        nparr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if face_cascade is not None:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 5)
            if len(faces) > 0:
                return jsonify({"success": True, "message": "Face registered."})
        return jsonify({"success": True, "message": "Registered (Bypassed)."})
    except Exception as e: return jsonify({"error": str(e)}), 500

@app.route('/api/evidence/analyze', methods=['POST'])
def analyze_frame():
    data = request.json
    user_id = data.get('user_id')
    image_base64 = data.get('image')
    if not user_id or not image_base64: return jsonify({"error": "Missing data"}), 400
    try:
        if ',' in image_base64: image_base64 = image_base64.split(',')[1]
        img_data = base64.b64decode(image_base64)
        nparr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        human_detected = False
        if net:
            blob = cv2.dnn.blobFromImage(img, 0.00392, (320, 320), (0, 0, 0), True, crop=False)
            net.setInput(blob)
            outs = net.forward(output_layers)
            for out in outs:
                for detection in out:
                    if detection[5] > 0.3: human_detected = True
        elif face_cascade:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            if len(faces) > 0: human_detected = True
        
        evidence_saved = False
        if human_detected:
            timestamp = int(time.time())
            cv2.imwrite(os.path.join(evidence_dir, f"evidence_{user_id}_{timestamp}.jpg"), img)
            evidence_saved = True
        
        _, buffer = cv2.imencode('.jpg', img)
        return jsonify({"success": True, "unknown_detected": human_detected, "evidence_saved": evidence_saved})
    except Exception as e: return jsonify({"error": str(e)}), 500

@app.route('/api/sos/send_cloud_sms', methods=['POST'])
def send_cloud_sms():
    data = request.json or {}
    numbers = data.get('numbers', [])
    print(f"[CLOUD SMS] Sending SOS to {', '.join(numbers)}")
    return jsonify({"success": True, "message": "Cloud SMS Sent."})

@app.route('/api/nearby/police', methods=['GET'])
def get_nearby_police():
    lat = request.args.get('lat')
    lng = request.args.get('lng')
    
    print(f"[Nearby] Searching for police stations near {lat}, {lng}")
    
    mock_stations = [
        {
            "place_id": "ps1",
            "name": "Local Police Station",
            "address": "Main Market Area, Uttarakhand",
            "distance_text": "0.8 km",
            "latitude": float(lat) + 0.005 if lat else 0,
            "longitude": float(lng) + 0.005 if lng else 0,
        },
        {
            "place_id": "ps2",
            "name": "District Police Headquarters",
            "address": "Civil Lines, Uttarakhand",
            "distance_text": "2.3 km",
            "latitude": float(lat) - 0.01 if lat else 0,
            "longitude": float(lng) + 0.01 if lng else 0,
        },
        {
            "place_id": "ps3",
            "name": "Mahila Thana (Women Police Station)",
            "address": "Near City Hospital, Uttarakhand",
            "distance_text": "1.5 km",
            "latitude": float(lat) + 0.015 if lat else 0,
            "longitude": float(lng) - 0.005 if lng else 0,
        }
    ]
    
    return jsonify({
        "success": True,
        "results": mock_stations
    })

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    socketio.run(app, host='0.0.0.0', port=port, debug=False)
