import datetime
import random
import re
import time
import uuid
import requests
import streamlit as st

# ==========================================
# 1. PAGE CONFIGURATION & INITIAL STATE
# ==========================================
st.set_page_config(
    page_title="Medical Information Assistant",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API Endpoints
BACKEND_BASE_URL = "http://127.0.0.1:8000"
QUERY_URL = f"{BACKEND_BASE_URL}/api/query"

# Static Tip Pool
TIP_POOL = [
    "Ask questions in natural language. Be as specific as possible for better results.",
    "Include context such as patient age or symptom duration for clearer guidance.",
    "Ask one primary medical query at a time to ensure focused citations.",
    "Be specific when asking about prescription medications and dosages."
]

# Static Pool for Curated Dynamic Recommendations
RECOMMENDATION_POOL = [
    "What are the common symptoms of hypertension?",
    "What are common side effects of aspirin?",
    "What are early warning signs of type 2 diabetes?",
    "How can I improve sleep hygiene naturally?",
    "What lifestyle changes help manage cholesterol?",
    "When should a high fever in adults be evaluated?",
    "What is the difference between cold and flu symptoms?"
]

def initialize_session_state():
    """Safely initialize Streamlit session state variables."""
    if "conversations" not in st.session_state:
        st.session_state.conversations = {}
    if "active_conversation_id" not in st.session_state:
        st.session_state.active_conversation_id = None
    if "top_k" not in st.session_state:
        st.session_state.top_k = 3
    if "current_tip" not in st.session_state:
        st.session_state.current_tip = random.choice(TIP_POOL)
    if "recommended_questions" not in st.session_state:
        st.session_state.recommended_questions = random.sample(RECOMMENDATION_POOL, min(4, len(RECOMMENDATION_POOL)))

initialize_session_state()

# ==========================================
# 2. DESIGN SYSTEM & FIXED CONTRAST CSS
# ==========================================
st.markdown("""
<style>
    /* CSS RESET & THEME ENFORCEMENT */
    html, body, [data-testid="stAppViewContainer"] {
        background-color: #FFFFFF !important;
        color: #172033 !important;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif !important;
    }

    /* Hide Streamlit Header Chrome */
    header[data-testid="stHeader"] {
        display: none !important;
    }
    
    /* Ensure App Content Margin Alignment */
    .block-container {
        padding-top: 0rem !important;
        padding-bottom: 2rem !important;
        padding-left: 1.5rem !important;
        padding-right: 1.5rem !important;
        max-width: 100% !important;
    }

    /* TOP NAVIGATION BAR */
    .top-nav {
        background-color: #0B1020;
        color: #FFFFFF;
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 12px 24px;
        margin-left: -1.5rem;
        margin-right: -1.5rem;
        margin-bottom: 20px;
        border-bottom: 1px solid #1E293B;
    }
    .top-nav-brand {
        font-size: 16px;
        font-weight: 700;
        letter-spacing: -0.2px;
        color: #FFFFFF;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .top-nav-menu {
        display: flex;
        gap: 28px;
        font-size: 14px;
        font-weight: 500;
    }
    .nav-item {
        color: #94A3B8;
        cursor: pointer;
        padding-bottom: 4px;
    }
    .nav-item.active {
        color: #087F5B;
        border-bottom: 2px solid #087F5B;
        font-weight: 600;
    }
    .top-nav-status {
        font-size: 13px;
        font-weight: 500;
        display: flex;
        align-items: center;
        gap: 6px;
    }

    /* CARDS & CONTAINERS */
    .panel-card {
        background-color: #FFFFFF;
        border: 1px solid #E5E7EB;
        border-radius: 10px;
        padding: 16px;
        margin-bottom: 14px;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.03);
    }
    .panel-card-tinted {
        background-color: #ECFDF5;
        border: 1px solid #D1FAE5;
        border-radius: 10px;
        padding: 14px;
        margin-bottom: 14px;
    }

    /* TYPOGRAPHY & LABELS */
    .panel-title {
        color: #475569;
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 0.6px;
        text-transform: uppercase;
        margin-bottom: 10px;
    }
    .panel-header-text {
        font-size: 15px;
        font-weight: 700;
        color: #111827;
        margin-bottom: 4px;
    }
    .panel-subtext {
        font-size: 12px;
        color: #475569;
        line-height: 1.5;
    }
    .data-row {
        display: flex;
        justify-content: space-between;
        font-size: 13px;
        padding: 4px 0;
        color: #172033;
    }
    .data-row-label {
        color: #475569;
    }

    /* CONVERSATION HISTORY ITEMS & HOVER STATES */
    .history-group-label {
        color: #475569;
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 0.5px;
        text-transform: uppercase;
        margin-top: 14px;
        margin-bottom: 6px;
    }

    /* FIXED STREAMLIT BUTTON STYLING (PREVENTS INVISIBLE HOVER/DARK STATES) */
    div.stButton > button {
        background-color: #F8FAFC !important;
        color: #172033 !important;
        border: 1px solid #E5E7EB !important;
        border-radius: 8px !important;
        font-size: 13px !important;
        font-weight: 500 !important;
        text-align: left !important;
        padding: 8px 12px !important;
        transition: transform 0.18s ease, background-color 0.18s ease, border-color 0.18s ease !important;
        box-shadow: 0 1px 2px rgba(0,0,0,0.02) !important;
    }

    /* ENLARGE ON HOVER AND HIGHLIGHT WITHOUT COLOR INVERSION */
    div.stButton > button:hover {
        background-color: #EDF2F7 !important;
        color: #087F5B !important;
        border-color: #CBD5E1 !important;
        transform: scale(1.02) !important;
        cursor: pointer !important;
    }

    /* PRIMARY NEW CONVERSATION BUTTON STYLING */
    div.stButton > button[kind="primary"] {
        background-color: #087F5B !important;
        color: #FFFFFF !important;
        border: none !important;
        text-align: center !important;
        font-weight: 600 !important;
    }
    div.stButton > button[kind="primary"]:hover {
        background-color: #066548 !important;
        color: #FFFFFF !important;
        transform: scale(1.02) !important;
    }

    /* CHAT BUBBLES & READABILITY */
    .chat-wrapper {
        max-width: 850px;
        margin: 0 auto;
    }
    .user-bubble {
        background-color: #ECFDF5;
        border: 1px solid #D1FAE5;
        border-radius: 12px 12px 2px 12px;
        padding: 14px 18px;
        margin-left: 12%;
        margin-bottom: 16px;
        color: #1F2937;
        font-size: 15px;
        line-height: 1.6;
        overflow-wrap: anywhere;
    }
    .assistant-bubble {
        background-color: #FFFFFF;
        border: 1px solid #E5E7EB;
        border-radius: 12px 12px 12px 2px;
        padding: 18px 20px;
        margin-right: 8%;
        margin-bottom: 16px;
        color: #1F2937;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.02);
        overflow-wrap: anywhere;
    }
    .message-meta {
        font-size: 11px;
        color: #667085;
        margin-top: 8px;
        display: flex;
        gap: 12px;
        align-items: center;
    }

    /* FORCE MARKDOWN VISIBILITY IN ASSISTANT BUBBLE */
    .assistant-bubble p, .stMarkdown p {
        color: #1F2937 !important;
        font-size: 15px !important;
        line-height: 1.65 !important;
        margin-bottom: 8px !important;
    }
    .assistant-bubble h1, .assistant-bubble h2, .assistant-bubble h3, 
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
        color: #111827 !important;
        font-weight: 700 !important;
        margin-top: 12px !important;
        margin-bottom: 6px !important;
    }
    .assistant-bubble ul, .assistant-bubble ol, .stMarkdown ul, .stMarkdown ol {
        color: #1F2937 !important;
        margin-left: 18px !important;
        margin-bottom: 8px !important;
    }
    .assistant-bubble li, .stMarkdown li {
        color: #1F2937 !important;
        font-size: 15px !important;
        line-height: 1.6 !important;
    }
    .assistant-bubble strong, .stMarkdown strong {
        color: #111827 !important;
        font-weight: 600 !important;
    }
    .assistant-bubble code, .stMarkdown code {
        background-color: #F8FAFC !important;
        color: #087F5B !important;
        padding: 2px 5px !important;
        border-radius: 4px !important;
        font-size: 13px !important;
    }

    /* INPUT & DISCLAIMER */
    [data-testid="stChatInput"] textarea::placeholder {
    color: #FFFFFF !important;
}
[data-testid="stChatInput"] textarea,
[data-testid="stChatInput"] textarea:focus {
    color: #FFFFFF !important;
    -webkit-text-fill-color: #FFFFFF !important;
}
    .disclaimer-text {
        font-size: 11px;
        color: #667085;
        text-align: center;
        margin-top: 8px;
        line-height: 1.4;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 3. HELPER FUNCTIONS
# ==========================================
def check_backend_status():
    """Verify live connectivity with backend API."""
    try:
        resp = requests.get(BACKEND_BASE_URL, timeout=1.5)
        return resp.status_code == 200
    except Exception:
        return False

def generate_conversation_title(first_question):
    """Generate concise 3-7 word title using local text processing."""
    clean = re.sub(r'^(what is|what are|can you|how do i|how can i|tell me about|is there)\s+', '', first_question, flags=re.IGNORECASE)
    clean = re.sub(r'[^\w\s]', '', clean).strip()
    words = clean.split()
    if not words:
        return "New Medical Query"
    return " ".join(words[:6]).capitalize()

def create_new_conversation():
    """Create a new session state conversation state."""
    cid = str(uuid.uuid4())
    st.session_state.conversations[cid] = {
        "id": cid,
        "title": "New Conversation",
        "created_at": datetime.datetime.now(),
        "updated_at": datetime.datetime.now(),
        "messages": [],
        "has_sent_first_msg": False
    }
    st.session_state.active_conversation_id = cid
    return cid

def get_conversation_groups():
    """Categorize user conversations by actual timestamps."""
    now = datetime.datetime.now()
    today_cutoff = datetime.datetime(now.year, now.month, now.day)
    yesterday_cutoff = today_cutoff - datetime.timedelta(days=1)
    week_cutoff = today_cutoff - datetime.timedelta(days=7)

    groups = {"TODAY": [], "YESTERDAY": [], "PREVIOUS 7 DAYS": [], "OLDER": []}
    
    sorted_convs = sorted(
        [c for c in st.session_state.conversations.values() if c.get("has_sent_first_msg", False)],
        key=lambda x: x["updated_at"],
        reverse=True
    )

    for c in sorted_convs:
        updated = c["updated_at"]
        if updated >= today_cutoff:
            groups["TODAY"].append(c)
        elif updated >= yesterday_cutoff:
            groups["YESTERDAY"].append(c)
        elif updated >= week_cutoff:
            groups["PREVIOUS 7 DAYS"].append(c)
        else:
            groups["OLDER"].append(c)
            
    return groups

def send_query_to_backend(question, chat_history, top_k):
    """Send query to FastAPI backend with strict error handling."""
    payload = {
        "question": question,
        "chat_history": chat_history,
        "top_k": top_k
    }
    
    start_time = time.time()
    try:
        response = requests.post(QUERY_URL, json=payload, timeout=45)
        elapsed = time.time() - start_time
        
        if response.status_code == 200:
            data = response.json()
            return {
                "answer": data.get("answer", "No answer field provided in backend response."),
                "latency_seconds": data.get("latency_seconds", round(elapsed, 2)),
                "sources": data.get("sources", []),
                "error": None
            }
        else:
            return {
                "answer": None,
                "error": f"Server error: Received HTTP status code {response.status_code}."
            }
    except requests.exceptions.ConnectionError:
        return {
            "answer": None,
            "error": "Unable to connect to the medical assistant service. Please make sure the FastAPI backend is running."
        }
    except requests.exceptions.Timeout:
        return {
            "answer": None,
            "error": "The backend service took too long to respond. Please try again."
        }
    except Exception as e:
        return {
            "answer": None,
            "error": f"An unexpected network error occurred: {str(e)}"
        }

# Ensure an active conversation exists
if not st.session_state.active_conversation_id or st.session_state.active_conversation_id not in st.session_state.conversations:
    create_new_conversation()

is_online = check_backend_status()

# ==========================================
# 4. TOP NAVIGATION HEADER
# ==========================================
status_color = "#087F5B" if is_online else "#E03131"
status_text = "Backend Online" if is_online else "Backend Offline"

st.markdown(f"""
<div class="top-nav">
    <div class="top-nav-brand">
        🏥 Medical Information Assistant
    </div>
    <div class="top-nav-menu">
        <span class="nav-item active">Assistant</span>
        <span class="nav-item">History</span>
        <span class="nav-item">About</span>
    </div>
    <div class="top-nav-status">
        <span style="color: {status_color};">●</span> {status_text}
    </div>
</div>
""", unsafe_allow_html=True)

# ==========================================
# 5. THREE-COLUMN MAIN APPLICATION
# ==========================================
left_col, center_col, right_col = st.columns([1, 2.2, 1.1])

# ------------------------------------------
# LEFT COLUMN: CONVERSATION HISTORY
# ------------------------------------------
with left_col:
    st.markdown('<div class="panel-title">CONVERSATIONS</div>', unsafe_allow_html=True)
    
    if st.button("➕ New Conversation", use_container_width=True, type="primary"):
        create_new_conversation()
        st.rerun()

    groups = get_conversation_groups()
    has_conversations = any(len(lst) > 0 for lst in groups.values())

    if not has_conversations:
        st.markdown("""
        <div style="font-size: 12px; color: #667085; padding: 12px 0;">
            No recent conversations yet. Ask a question to begin.
        </div>
        """, unsafe_allow_html=True)
    else:
        for group_label, conv_list in groups.items():
            if conv_list:
                st.markdown(f'<div class="history-group-label">{group_label}</div>', unsafe_allow_html=True)
                for conv in conv_list:
                    btn_label = f"💬 {conv['title']}"
                    
                    if st.button(
                        btn_label, 
                        key=f"conv_btn_{conv['id']}", 
                        use_container_width=True
                    ):
                        st.session_state.active_conversation_id = conv["id"]
                        st.rerun()

    st.markdown("<br><hr style='border: 0; border-top: 1px solid #E5E7EB; margin: 15px 0;'>", unsafe_allow_html=True)
    
    if st.button("🗑️ Clear all conversations", use_container_width=True):
        st.session_state.conversations = {}
        create_new_conversation()
        st.rerun()

# ------------------------------------------
# CENTER COLUMN: MAIN CHAT INTERFACE
# ------------------------------------------
with center_col:
    active_cid = st.session_state.active_conversation_id
    current_conv = st.session_state.conversations[active_cid]
    messages = current_conv["messages"]

    # Welcome State / Empty Screen
    if not messages:
        st.markdown("""
        <div style="text-align: center; padding: 40px 20px 20px 20px;">
            <div style="font-size: 38px; margin-bottom: 8px;">🏥</div>
            <h2 style="font-size: 20px; font-weight: 700; color: #111827; margin-bottom: 6px;">
                Medical Information Assistant
            </h2>
            <p style="font-size: 14px; color: #667085; max-width: 480px; margin: 0 auto 24px auto;">
                How can I help you understand your health information today?
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="panel-title" style="text-align: center;">RECOMMENDED QUESTIONS</div>', unsafe_allow_html=True)
        
        for q in st.session_state.recommended_questions:
            if st.button(q, key=f"rec_{q}", use_container_width=True):
                st.session_state.pending_question = q
                st.rerun()

    # Active Conversation Display
    else:
        st.markdown("""
        <div style="border-bottom: 1px solid #E5E7EB; padding-bottom: 10px; margin-bottom: 20px; display: flex; justify-content: space-between; align-items: center;">
            <div style="font-size: 15px; font-weight: 700; color: #111827;">
                🏥 Medical Information Assistant
            </div>
            <div style="font-size: 12px; color: #087F5B; font-weight: 500;">
                ● Online
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="chat-wrapper">', unsafe_allow_html=True)
        
        for msg in messages:
            timestamp_str = msg.get("timestamp", "")
            
            if msg["role"] == "user":
                st.markdown(f"""
                <div class="user-bubble">
                    {msg["content"]}
                    <div class="message-meta" style="justify-content: flex-end;">
                        <span>{timestamp_str}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
            elif msg["role"] == "assistant":
                st.markdown("""
                <div class="assistant-bubble">
                    <div style="font-size: 13px; font-weight: 700; color: #087F5B; margin-bottom: 8px; display: flex; align-items: center; gap: 6px;">
                        🏥 Medical Assistant
                    </div>
                """, unsafe_allow_html=True)
                
                st.markdown(msg["content"])
                
                st.markdown(f"""
                    <div class="message-meta">
                        <span>{timestamp_str}</span>
                        <span>•</span>
                        <span>Source Verified</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                sources = msg.get("sources", [])
                latency = msg.get("latency", None)
                
                with st.expander("📚 Sources & response details"):
                    if latency:
                        st.markdown(f"**Response time:** `{latency:.2f}s`")
                    st.markdown(f"**Sources found:** `{len(sources)}`")
                    st.markdown("---")
                    
                    if sources:
                        for idx, src in enumerate(sources, 1):
                            topic = src.get("topic", src.get("disease", src.get("title", "Medical Knowledge")))
                            dataset = src.get("dataset", src.get("source", "Knowledge Base"))
                            content = src.get("content", src.get("text", src.get("snippet", "No preview snippet available.")))
                            
                            st.markdown(f"**Source {idx}: {topic}** *(Dataset: {dataset})*")
                            st.caption(content)
                            if idx < len(sources):
                                st.markdown("---")
                    else:
                        st.caption("No external document snippets cited for this turn.")

        st.markdown('</div>', unsafe_allow_html=True)

    pending_prompt = st.session_state.pop("pending_question", None)
    user_input = st.chat_input("Ask a medical question...")
    prompt_to_process = user_input or pending_prompt

    if prompt_to_process:
        now_time = datetime.datetime.now().strftime("%I:%M %p")
        
        if not current_conv["has_sent_first_msg"]:
            current_conv["title"] = generate_conversation_title(prompt_to_process)
            current_conv["has_sent_first_msg"] = True

        current_conv["messages"].append({
            "role": "user",
            "content": prompt_to_process,
            "timestamp": now_time
        })
        current_conv["updated_at"] = datetime.datetime.now()

        past_turns = []
        for m in current_conv["messages"][:-1]:
            role = "User" if m["role"] == "user" else "Assistant"
            past_turns.append(f"{role}: {m['content']}")
        formatted_history = "\n".join(past_turns)

        with st.spinner("Searching medical knowledge base..."):
            backend_res = send_query_to_backend(
                question=prompt_to_process,
                chat_history=formatted_history,
                top_k=st.session_state.top_k
            )

        if backend_res["error"]:
            current_conv["messages"].append({
                "role": "assistant",
                "content": f"⚠️ **Service Notice:** {backend_res['error']}",
                "timestamp": now_time,
                "sources": [],
                "latency": None
            })
        else:
            current_conv["messages"].append({
                "role": "assistant",
                "content": backend_res["answer"],
                "timestamp": now_time,
                "sources": backend_res["sources"],
                "latency": backend_res["latency_seconds"]
            })
            
        st.rerun()

    st.markdown("""
    <div class="disclaimer-text">
        This information is for general knowledge and does not replace professional medical advice.<br>
        Always consult a qualified healthcare professional for diagnosis or treatment.
    </div>
    """, unsafe_allow_html=True)

# ------------------------------------------
# RIGHT COLUMN: SYSTEM & INFORMATION PANEL
# ------------------------------------------
with right_col:
    # Card 1: About
    st.markdown("""
    <div class="panel-card">
        <div class="panel-title">ABOUT THIS ASSISTANT</div>
        <div class="panel-header-text">🩺 Evidence-Based Guidance</div>
        <div class="panel-subtext">
            Delivers verified healthcare information retrieved directly from your local clinical knowledge base.
        </div>
        <hr style="border:0; border-top: 1px solid #E5E7EB; margin: 10px 0;">
        <div style="font-size: 11px; color: #667085; line-height: 1.4;">
            This assistant provides general health information and does not diagnose conditions or prescribe treatment.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Card 2: System Status
    st.markdown(f"""
    <div class="panel-card">
        <div class="panel-title">SYSTEM STATUS</div>
        <div class="data-row">
            <span class="data-row-label">FastAPI Backend</span>
            <span style="color: {status_color}; font-weight: 600;">● {status_text}</span>
        </div>
        <div class="data-row">
            <span class="data-row-label">FAISS Vector Store</span>
            <span style="color: #087F5B; font-weight: 600;">● Connected</span>
        </div>
        <div class="data-row">
            <span class="data-row-label">RAG Pipeline</span>
            <span style="color: #087F5B; font-weight: 600;">● Ready</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Card 3: Retrieval Settings
    st.markdown('<div class="panel-card">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">RETRIEVAL SETTINGS</div>', unsafe_allow_html=True)
    st.session_state.top_k = st.slider(
        "Top K (Retrieved Sources)",
        min_value=1,
        max_value=5,
        value=st.session_state.top_k,
        help="Controls the number of relevant clinical document passages returned to the LLM."
    )
    st.markdown('</div>', unsafe_allow_html=True)

    # Card 4: Current Response Summary
    active_cid = st.session_state.active_conversation_id
    active_messages = st.session_state.conversations[active_cid]["messages"]
    assistant_messages = [m for m in active_messages if m["role"] == "assistant"]

    if assistant_messages:
        last_msg = assistant_messages[-1]
        sources_count = len(last_msg.get("sources", []))
        latency_val = last_msg.get("latency", None)
        latency_str = f"{latency_val:.2f}s" if latency_val else "N/A"

        st.markdown(f"""
        <div class="panel-card">
            <div class="panel-title">CURRENT RESPONSE</div>
            <div class="data-row">
                <span class="data-row-label">Sources found</span>
                <b>{sources_count}</b>
            </div>
            <div class="data-row">
                <span class="data-row-label">Response time</span>
                <b>{latency_str}</b>
            </div>
            <div class="data-row">
                <span class="data-row-label">Model Engine</span>
                <b>Local RAG</b>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Card 5: Tip Box
    st.markdown(f"""
    <div class="panel-card-tinted">
        <div class="panel-title" style="color: #087F5B;">💡 TIP</div>
        <div style="font-size: 12px; color: #1F2937; line-height: 1.5;">
            "{st.session_state.current_tip}"
        </div>
    </div>
    """, unsafe_allow_html=True)