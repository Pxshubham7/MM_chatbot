import base64
import json
import os
import requests
import streamlit as st

# ==============================================================================
# Configuration & Backend Endpoint
# (Streamlit connects ONLY to the FastAPI backend; zero direct database access)
# ==============================================================================
API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")

st.set_page_config(
    page_title="Mindful Companion - Non-Clinical Emotional Support",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for polished, calming appearance
st.markdown("""
<style>
    .reportview-container {
        background: #fdfbf7;
    }
    .disclaimer-box {
        background-color: #fff8e6;
        border-left: 4px solid #ffb020;
        padding: 12px 16px;
        border-radius: 4px;
        margin-bottom: 15px;
        font-size: 0.88rem;
        color: #5c4700;
    }
    .privacy-box {
        background-color: #eef6ff;
        border-left: 4px solid #1e88e5;
        padding: 12px 16px;
        border-radius: 4px;
        margin-bottom: 20px;
        font-size: 0.88rem;
        color: #0d47a1;
    }
    .crisis-box {
        background-color: #ffebee;
        border-left: 4px solid #e53935;
        padding: 14px 18px;
        border-radius: 4px;
        margin-bottom: 15px;
        font-size: 0.92rem;
        color: #b71c1c;
    }
    .citation-card {
        background-color: #f5f5f5;
        border-radius: 6px;
        padding: 10px 14px;
        margin-top: 8px;
        border: 1px solid #e0e0e0;
        font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)


# ==============================================================================
# Ephemeral Session State Initialization
# (Conversations exist only in client browser memory; never persisted to DB)
# ==============================================================================
if "messages" not in st.session_state:
    st.session_state.messages = []


def check_backend_health():
    """Checks FastAPI backend connectivity and RAG status."""
    try:
        res = requests.get(f"{API_BASE_URL}/health", timeout=3.0)
        if res.status_code == 200:
            return res.json()
        return {"status": "unhealthy", "error": f"Status code {res.status_code}"}
    except Exception as exc:
        return {"status": "unreachable", "error": str(exc)}


# ==============================================================================
# Sidebar: System Controls & Privacy Disclaimers
# ==============================================================================
with st.sidebar:
    st.title("🌿 Mindful Companion")
    st.caption("A privacy-conscious, non-clinical emotional support guide.")

    # Backend Connectivity Indicator
    health_data = check_backend_health()
    if health_data.get("status") in ["healthy", "degraded"]:
        st.success(f"Backend Connected ({health_data.get('database_type', 'db').upper()})")
        st.caption(f"Curated Knowledge Documents: {health_data.get('documents_indexed', 0)}")
    else:
        st.error(f"Backend Offline ({API_BASE_URL})")
        st.caption("Please ensure FastAPI is running.")

    st.divider()

    # Session Controls
    st.subheader("Session Settings")
    generate_speech = st.checkbox("Generate Spoken Audio (TTS)", value=False, help="Receive optional voice response audio.")

    if st.button("🧹 Clear Conversation History"):
        st.session_state.messages = []
        st.rerun()

    st.divider()

    # Security & Privacy Guidelines
    st.subheader("Security & Privacy")
    st.markdown("""
    - **Single Trusted User:** Designed for local machine or private network access.
    - **Ephemeral Retention:** Messages and media are held in this active tab only. Nothing is saved to permanent databases or logs.
    - **No End-to-End Encryption:** The server processes content in temporary memory to formulate responses.
    - **Strict Safeguards:** Retrieved documents provide evidence but cannot override ethical safety rules.
    """)


# ==============================================================================
# Main Chat Area
# ==============================================================================
st.markdown("### Mindful Companion")

# Prominent Ethical Disclaimers
st.markdown("""
<div class="disclaimer-box">
    <strong>⚠️ Non-Clinical Conversational Guide:</strong> This application is designed for everyday reflection,
    mindfulness, and non-clinical emotional support. It is <strong>not</strong> a therapist, doctor, or medical provider,
    and cannot diagnose conditions, prescribe treatments, or provide psychiatric care.
</div>
<div class="privacy-box">
    <strong>🔒 Ephemeral Session:</strong> Your chat messages, uploaded images, and audio clips are processed in temporary memory and are <strong>never stored</strong> in any database or disk log.
</div>
""", unsafe_allow_html=True)


# Render Chat History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg.get("is_crisis"):
            st.markdown(f'<div class="crisis-box">{msg["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(msg["content"])

        # Display Media Annotations if present
        if msg.get("image_description"):
            st.caption(f"🖼️ *{msg['image_description']}*")
        if msg.get("transcribed_text"):
            st.caption(f"🎙️ *Transcribed Audio:* \"{msg['transcribed_text']}\"")

        # Display Grounded Citations
        citations = msg.get("citations", [])
        if citations:
            with st.expander(f"📚 Grounded Source Citations ({len(citations)})"):
                for cit in citations:
                    st.markdown(f"""
                    <div class="citation-card">
                        <strong>Source:</strong> {cit['document_title']} (<em>Category: {cit['category']}</em>)<br>
                        <strong>Excerpt:</strong> "{cit['content_excerpt']}"<br>
                        <small>Similarity Confidence: {cit['similarity_score'] * 100:.1f}% | Path: <code>{cit['source_uri']}</code></small>
                    </div>
                    """, unsafe_allow_html=True)
        elif msg["role"] == "assistant" and not msg.get("is_crisis"):
            st.caption("ℹ️ *Response based on general reflective support (topic not covered in curated documents).*")

        # Play Spoken Audio if present
        if msg.get("audio_base64"):
            audio_bytes = base64.b64decode(msg["audio_base64"])
            st.audio(audio_bytes, format="audio/wav")


# ==============================================================================
# Multimodal Uploads & Input Form
# ==============================================================================
with st.expander("📎 Optional Media Uploads (Image / Audio)", expanded=False):
    col1, col2 = st.columns(2)
    with col1:
        uploaded_image = st.file_uploader(
            "Upload an image (JPG, PNG)",
            type=["jpg", "jpeg", "png", "webp"],
            help="Upload a calming picture, journal photo, or drawing (Max 10MB)"
        )
        if uploaded_image:
            st.image(uploaded_image, width=200, caption="Attached Image")
    with col2:
        uploaded_audio = st.file_uploader(
            "Upload an audio clip (WAV, MP3)",
            type=["wav", "mp3", "ogg"],
            help="Upload a voice reflection to be transcribed (Max 25MB)"
        )
        if uploaded_audio:
            st.audio(uploaded_audio)

user_text = st.chat_input("Share what is on your mind, or ask about coping techniques...")

if user_text or uploaded_image or uploaded_audio:
    display_user_text = user_text or "(Uploaded media reflection)"
    
    # Append user turn to local ephemeral state
    st.session_state.messages.append({
        "role": "user",
        "content": display_user_text
    })

    # Prepare multipart request to FastAPI
    files = {}
    if uploaded_image is not None:
        files["image_file"] = (uploaded_image.name, uploaded_image.getvalue(), uploaded_image.type)
    if uploaded_audio is not None:
        files["audio_file"] = (uploaded_audio.name, uploaded_audio.getvalue(), uploaded_audio.type)

    # Format ephemeral dialogue history for context window
    history_payload = [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state.messages[:-1]
    ]

    form_data = {
        "message": user_text or "",
        "conversation_history_json": json.dumps(history_payload),
        "generate_speech": str(generate_speech).lower()
    }

    with st.spinner("Reflecting and consulting curated wellness guidance..."):
        try:
            response = requests.post(
                f"{API_BASE_URL}/api/chat",
                data=form_data,
                files=files if files else None,
                timeout=45.0
            )

            if response.status_code == 200:
                data = response.json()
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": data["answer"],
                    "citations": data.get("citations", []),
                    "has_rag_context": data.get("has_rag_context", False),
                    "is_crisis": data.get("is_crisis_response", False),
                    "audio_base64": data.get("audio_base64"),
                    "transcribed_text": data.get("transcribed_text"),
                    "image_description": data.get("image_description")
                })
            else:
                err_detail = response.json().get("detail", response.text)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"⚠️ *Error communicating with service ({response.status_code}):* {err_detail}",
                    "is_crisis": False
                })
        except requests.exceptions.RequestException as exc:
            st.session_state.messages.append({
                "role": "assistant",
                "content": f"⚠️ *Connection Error:* Unable to reach backend server at `{API_BASE_URL}`. ({exc})",
                "is_crisis": False
            })

    st.rerun()
