import streamlit as st
from datetime import datetime
import json
import os
import uuid
import time
import google.generativeai as genai

# Replace with your actual Gemini API key
GEMINI_API_KEY = "AIzaSyDhMC_PEi-3ueM7a6jVc1qZhxTQSfQd7ZU"
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-2.0-flash')

def get_chat_store():
    if os.path.exists("chat_store.json"):
        with open("chat_store.json", "r") as f:
            return json.load(f)
    return {"rooms": {}}

def save_chat_store(store):
    with open("chat_store.json", "w") as f:
        json.dump(store, f)

def create_chat_room(case_id, creator_name, case_description):
    store = get_chat_store()
    if case_id not in store["rooms"]:
        room_data = {
            "id": case_id,
            "created_at": datetime.now().isoformat(),
            "creator": creator_name,
            "description": case_description,
            "participants": [creator_name, "Dr. AI Assistant"],
            "messages": []
        }
        welcome_message = {
            "id": str(uuid.uuid4()),
            "user": "Dr. AI Assistant",
            "content": f"Welcome to the medical consultation for '{case_description}'I'm Dr. AI, your virtual medical expert. Feel free to ask any health-related or report-related questions.",
            "type": "text",
            "timestamp": datetime.now().isoformat()
        }
        room_data["messages"].append(welcome_message)
        store["rooms"][case_id] = room_data
        save_chat_store(store)
    return case_id

def create_manual_chat_room(creator_name, case_description):
    case_id = f"MANUAL-{datetime.now().strftime('%Y%m%d%H%M%S')}"
    return create_chat_room(case_id, creator_name, case_description)

def join_chat_room(case_id, user_name):
    store = get_chat_store()
    if case_id in store["rooms"]:
        if user_name not in store["rooms"][case_id]["participants"]:
            store["rooms"][case_id]["participants"].append(user_name)
            save_chat_store(store)
        return True
    return False

def add_message(case_id, user_name, message, message_type="text"):
    store = get_chat_store()
    if case_id in store["rooms"]:
        message_data = {
            "id": str(uuid.uuid4()),
            "user": user_name,
            "content": message,
            "type": message_type,
            "timestamp": datetime.now().isoformat()
        }
        store["rooms"][case_id]["messages"].append(message_data)
        save_chat_store(store)
        return message_data
    return None

def get_messages(case_id):
    store = get_chat_store()
    if case_id in store["rooms"]:
        return store["rooms"][case_id]["messages"]
    return []

def get_available_rooms():
    store = get_chat_store()
    rooms = []
    for room_id, room_data in store["rooms"].items():
        rooms.append({
            "id": room_id,
            "description": room_data["description"],
            "creator": room_data["creator"],
            "created_at": room_data["created_at"],
            "participants": len(room_data["participants"])
        })
    rooms.sort(key=lambda x: x["created_at"], reverse=True)
    return rooms

def get_gemini_response(user_question, case_description, findings=None):
    findings_text = ""
    if findings and len(findings) > 0:
        findings_text = "The uploaded report shows the following findings:\n"
        for i, finding in enumerate(findings, 1):
            findings_text += f"{i}. {finding}\n"

    system_prompt = f"""
You are Dr. MediBot, a highly advanced and expert virtual medical doctor specializing in providing clear, accurate, and comprehensive medical information.

Your core responsibilities are:

1.  **Precise Medical Explanations:** Explain diseases, symptoms, medications, medical procedures, and diagnostic results in simple, easy-to-understand language. Break down complex medical jargon into digestible insights.
2.  **Medical Report Interpretation:** When presented with a medical report (e.g., lab results, imaging reports, pathology findings), meticulously interpret the data, highlight key findings, explain their significance, and contextualize them for the user.
3.  **General Health Queries:** Answer a broad range of general health and wellness questions with professional expertise and helpful guidance.
4.  **Symptom Understanding (Not Diagnosis):** If a user describes symptoms, you will explain what those symptoms could indicate (common conditions, relevant body systems) and emphasize the importance of professional medical consultation for diagnosis. You *will not* provide a diagnosis or recommend specific treatments.
5.  **Medication Information:** Provide details about medications, including their uses, common side effects, and important considerations. *Do not* prescribe or recommend starting/stopping any medication.
6.  **Professional Demeanor:** Maintain a consistently clear, empathetic, helpful, and highly professional tone in all interactions. Avoid jargon where simpler terms suffice.
7.  **Safety Disclaimer:** Always include a crucial disclaimer at the end of every response: "Please remember, I am an AI and cannot replace a qualified medical professional. Always consult with a doctor for diagnosis, treatment, and personalized medical advice."

Your primary goal is to empower users with accurate medical knowledge, facilitate their understanding of health-related topics, and guide them towards appropriate professional medical care when necessary.

Case: {case_description}
{findings_text}
"""

    try:
        response = model.generate_content(
            [
                {"role": "user", "parts": [system_prompt]},
                {"role": "user", "parts": [user_question]}
            ],
            generation_config=genai.types.GenerationConfig(
                max_output_tokens=500,
                temperature=0.4
            )
        )
        return response.text
    except Exception as e:
        return f"I encountered an error while analyzing your query: {e}"

def render_chat_interface():
    st.subheader("🩺 Medical Expert Chatbot")

    if "user_name" not in st.session_state:
        st.session_state.user_name = "Patient"

    user_name = st.text_input("Your Name", value=st.session_state.user_name)
    if user_name != st.session_state.user_name:
        st.session_state.user_name = user_name

    tab1, tab2, tab3 = st.tabs(["Join Case", "Create from Report", "Start Chat"])

    with tab1:
        rooms = get_available_rooms()
        if rooms:
            room_options = {f"{r['id']} - {r['description']} (by {r['creator']})": r['id'] for r in rooms}
            selected = st.selectbox("Select a Case", list(room_options.keys()))
            if st.button("Join Chat"):
                cid = room_options[selected]
                if join_chat_room(cid, user_name):
                    st.session_state.current_case_id = cid
                    st.rerun()
        else:
            st.info("No cases available.")

    with tab2:
        case_description = st.text_input("Case Description")
        can_create = st.session_state.get("file_type") and st.session_state.get("file_data")
        if can_create:
            case_id = f"{st.session_state.file_type.upper()}-{datetime.now().strftime('%Y%m%d%H%M%S')}"
        else:
            case_id = f"REPORT-{datetime.now().strftime('%Y%m%d%H%M%S')}"

        if st.button("Start Case Chat"):
            if case_description:
                cid = create_chat_room(case_id, user_name, case_description)
                st.session_state.current_case_id = cid
                st.rerun()
            else:
                st.error("Please enter a description.")

    with tab3:
        case_description = st.text_area("Describe your problem or symptoms")
        if st.button("Start Chat"):
            if case_description:
                cid = create_manual_chat_room(user_name, case_description)
                st.session_state.current_case_id = cid
                st.rerun()
            else:
                st.error("Please provide a case description.")

    if "current_case_id" in st.session_state:
        cid = st.session_state.current_case_id
        store = get_chat_store()
        if cid in store["rooms"]:
            room = store["rooms"][cid]
            st.subheader(f"Case: {room['description']}")
            st.caption(f"By {room['creator']} | Participants: {len(room['participants'])}")
            messages = get_messages(cid)
            for m in messages:
                with st.chat_message(name=m["user"], avatar="👨‍⚕️"):
                    st.write(m["content"])

            q = st.chat_input("Ask a medical question")
            if q:
                add_message(cid, user_name, q)
                with st.spinner("Dr. AI is responding..."):
                    findings = st.session_state.get("findings")
                    ans = get_gemini_response(q, room['description'], findings)
                    add_message(cid, "Dr. AI Assistant", ans)
                st.rerun()
        else:
            st.error("Case not found.")
            if st.button("Go Back"):
                del st.session_state.current_case_id
                st.rerun()

if __name__ == "__main__":
    st.title("🤖 AI Medical Chatbot & Report Advisor")
    render_chat_interface()
