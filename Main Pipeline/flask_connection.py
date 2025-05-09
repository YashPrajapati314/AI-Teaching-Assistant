from flask import Flask, request, jsonify
from main_chat import ChatSession

app = Flask(__name__)
sessions: dict[str, ChatSession] = {}

@app.route("/start_chat", methods=["POST"])
def start_chat():
    session_id = request.json["session_id"]
    sessions[session_id] = ChatSession()
    return jsonify({"message": "New session started."})

@app.route("/send_message", methods=["POST"])
def send_message():
    session_id = request.json["session_id"]
    user_message = request.json["message"]
    
    if session_id not in sessions:
        return jsonify({"error": "Session not found"}), 404

    teacher_response = sessions[session_id].send_message(user_message)
    return jsonify({"response": teacher_response})
