import firebase_admin
from firebase_admin import firestore

class RakshaFirebaseService:
    def __init__(self):
        try:
            if firebase_admin._apps:
                self.db = firestore.client()
                print("[Firebase] Firestore client initialized successfully.")
            else:
                self.db = None
                print("[Firebase] Firestore unavailable (Firebase admin app not initialized).")
        except Exception as e:
            self.db = None
            print(f"[Firebase] Error initializing Firestore client: {e}")

    def save_chat_message(self, user_id, message):
        if not self.db:
            print("[Firebase] Firestore unavailable - message not saved.")
            return False
        try:
            data = {}
            if isinstance(message, dict):
                data = message.copy()
            else:
                data = {"message": str(message)}

            data['timestamp'] = firestore.SERVER_TIMESTAMP

            self.db.collection("users").document(user_id).collection("chat_history").add(data)
            print("[Firebase] Chat saved successfully.")
            return True
        except Exception as e:
            print(f"[Firebase] Error saving chat message: {e}")
            return False

    def save_emergency_log(self, data):
        if not self.db:
            print("[Firebase] Firestore unavailable - emergency log not saved.")
            return False
        try:
            log_data = data.copy() if isinstance(data, dict) else {"log": str(data)}
            log_data['timestamp'] = firestore.SERVER_TIMESTAMP

            self.db.collection("emergency_logs").add(log_data)
            print("[Firebase] Emergency log saved successfully.")
            return True
        except Exception as e:
            print(f"[Firebase] Error saving emergency log: {e}")
            return False

    def save_evidence(self, user_id, image_url):
        if not self.db:
            print("[Firebase] Firestore unavailable - evidence not saved.")
            return False
        try:
            evidence_data = {
                "imageUrl": image_url,
                "timestamp": firestore.SERVER_TIMESTAMP
            }
            self.db.collection("users").document(user_id).collection("evidence").add(evidence_data)
            print("[Firebase] Evidence saved successfully.")
            return True
        except Exception as e:
            print(f"[Firebase] Error saving evidence: {e}")
            return False

    def get_user(self, user_id):
        if not self.db:
            print("[Firebase] Firestore unavailable - cannot get user.")
            return None
        try:
            doc = self.db.collection("users").document(user_id).get()
            if doc.exists:
                return doc.to_dict()
            else:
                print(f"[Firebase] User document {user_id} not found.")
                return None
        except Exception as e:
            print(f"[Firebase] Error retrieving user {user_id}: {e}")
            return None
