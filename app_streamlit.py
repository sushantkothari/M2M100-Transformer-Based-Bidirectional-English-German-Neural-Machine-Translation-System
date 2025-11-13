# app_streamlit.py
import streamlit as st
import requests

# ---------- Configuration ----------
API_URL = "http://127.0.0.1:8000/translate"  # FastAPI backend
st.set_page_config(page_title="EN ↔ DE Translator", layout="wide")

# ---------- Header ----------
st.markdown(
    """
    <h1 style='text-align: center; color: #4B8BBE;'>English ↔ German Translator</h1>
    <p style='text-align: center; color: #555; font-size: 18px;'>
    Translate text between English and German using your local fine-tuned M2M100 model.
    </p>
    <hr style='height:2px;border:none;color:#333;background-color:#333;' />
    """,
    unsafe_allow_html=True
)

# ---------- Language Selection ----------
col1, col2 = st.columns([1, 1])
with col1:
    src_lang = st.selectbox("Source Language", ["en", "de"], index=0)
with col2:
    tgt_lang = st.selectbox("Target Language", ["de", "en"], index=1 if src_lang == "en" else 0)

st.markdown("---")

# ---------- Text Input ----------
text = st.text_area(
    "Enter text to translate:",
    height=200,
    placeholder="Type English or German text here..."
)

# ---------- Session State ----------
if "last_input" not in st.session_state:
    st.session_state["last_input"] = ""
if "last_output" not in st.session_state:
    st.session_state["last_output"] = []

# ---------- Translate Button ----------
if st.button("🚀 Translate") and text.strip():
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    if not lines:
        st.warning("Please enter at least one line of text.")
    else:
        # Reuse cached output
        if text == st.session_state["last_input"] and st.session_state["last_output"]:
            translations = st.session_state["last_output"]
        else:
            payload = {"texts": lines, "src_lang": src_lang, "tgt_lang": tgt_lang, "num_beams": 2}
            try:
                with st.spinner("Translating... This may take a few seconds on CPU."):
                    r = requests.post(API_URL, json=payload, timeout=180)
                    r.raise_for_status()
                    translations = r.json().get("translations", [])
                st.session_state["last_input"] = text
                st.session_state["last_output"] = translations
            except requests.exceptions.RequestException as e:
                st.error(f"API error: {e}")
                translations = []

        # ---------- Display Translations ----------
        if translations:
            for idx, (src, tgt) in enumerate(zip(lines, translations), 1):
                st.markdown(f"### ✏️ Source {idx}")
                st.info(src)
                st.markdown(f"### ✅ Translation {idx}")
                st.success(tgt)
                st.markdown("---")
        else:
            st.info("No translations returned. Make sure your FastAPI backend is running.")

# ---------- Footer ----------
st.markdown(
    """
    <p style='text-align: center; color: #888; font-size: 14px; margin-top: 50px;'>
    Powered by local fine-tuned M2M100 model | FastAPI + Streamlit
    </p>
    """,
    unsafe_allow_html=True
)
