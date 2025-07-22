import streamlit as st
import google.generativeai as genai
import json
import os
import uuid
import numpy as np
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity

# Medical Chatbot System
class MedicalChatbot:
    def __init__(self):
        self.api_key = 'AIzaSyDhMC_PEi-3ueM7a6jVc1qZhxTQSfQd7ZU'
        self.conversation_history = []
        self.analysis_store = self.load_analysis_store()
        self.model = self.initialize_model()

    def initialize_model(self):
        genai.configure(api_key=self.api_key)
        return genai.GenerativeModel(
            model_name="gemini-2.0-flash",
            system_instruction=(
                "You are a friendly and knowledgeable medical assistant. "
                "Always respond in both English and Hindi. Answer patient queries with empathy. "
                "Simplify medical terminology where possible. "
                "Be professional but understandable by someone with no medical background."
            )
        )

    def load_analysis_store(self):
        if os.path.exists("analysis_store.json"):
            with open("analysis_store.json", "r") as f:
                return json.load(f)
        return {"analyses": []}

    def get_embeddings(self, text):
        return np.random.rand(1536)  # Simulated

    def get_relevant_contexts(self, query, top_k=3):
        query_embedding = self.get_embeddings(query)
        analyses = self.analysis_store["analyses"]
        contexts = []

        if not analyses:
            return ["No previous analyses found."]

        for analysis in analyses:
            analysis_text = analysis.get("analysis", "")
            findings_text = "\n".join(f"- {f}" for f in analysis.get("findings", []))
            full_text = f"{analysis_text}\n\nFindings:\n{findings_text}\nImage: {analysis.get('filename')}\nDate: {analysis.get('date', '')[:10]}"
            contexts.append({
                "text": full_text,
                "embedding": self.get_embeddings(full_text),
                "id": analysis.get("id", ""),
                "date": analysis.get("date", "")
            })

        similarities = [(cosine_similarity([query_embedding], [ctx["embedding"]])[0][0], ctx) for ctx in contexts]
        top_contexts = [ctx["text"] for _, ctx in sorted(similarities, reverse=True)[:top_k]]
        return top_contexts

    def analyze_report_content(self, report_content):
        prompt = (
            f"Please analyze this medical report:\n\n{report_content}\n\n"
            "Give a simple summary of the report in both English and Hindi. "
            "Explain the issues clearly, suggest precautions and next steps. "
            "If there are no major problems, mention that clearly in both languages."
        )
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            print(f"Error: {e}")
            return "⚠️ Sorry, an error occurred while analyzing the report."

    def chat_with_user(self, user_input):
        try:
            response = self.model.generate_content(user_input)
            return response.text
        except Exception as e:
            return "⚠️ Sorry, an error occurred while processing your question."

# Streamlit UI
st.set_page_config(page_title="Medical Chatbot", layout="centered")
st.title("🩺 Medical Report Chatbot (Hindi + English)")

qa_system = MedicalChatbot()

# Section 1: Analyze Report
st.subheader("📄 Analyze Medical Report")
report_content = st.text_area("Paste the medical report here:", height=200)

if st.button("🔍 Analyze Report"):
    if report_content.strip():
        with st.spinner("Analyzing your medical report..."):
            result = qa_system.analyze_report_content(report_content)
            st.success("✅ Report Analysis Completed")
            st.write(result)
    else:
        st.error("Please provide a medical report.")

# Section 2: Medical Chatbot (Conversational)
st.subheader("💬 Ask Questions About Your Health (बोलें या लिखें)")

user_input = st.text_input("Ask any question related to your health or the report (in Hindi or English):")

if st.button("🗨️ Ask"):
    if user_input.strip():
        with st.spinner("Getting response..."):
            reply = qa_system.chat_with_user(user_input)
            st.success("✅ Response Received")
            st.markdown(reply)
    else:
        st.warning("Please ask a question.")


