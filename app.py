import streamlit as st
from PIL import Image
import pandas as pd
from certificate_scores_clean_14_1_3 import classify_certificate, extract_score, filter_universities

# Page setup
st.set_page_config(page_title="University Matcher", layout="wide", page_icon="🎓")

# Multilang
texts = {
    "en": {
        "title": "🎓 Upload your certificates – see matched universities",
        "upload": "",
        "uploaded": "{} images uploaded.",
        "cert_type": "✅ <span style='color:#2ecc71'><b>Certificate type:</b> {}</span> (<b>Accuracy:</b> {:.1f}%), <b>Score:</b> {}",
        "cert_not_found": "Score not detected for {}.",
        "no_scores": "⚠️ No certificate scores detected.",
        "matched_universities": "<h4 style='color:#3498db'>🎯 Matched universities based on your certificates:</h4>",
        "no_match": "⚠️ Sorry, no universities matched your certificate scores.",
    },
    "uz": {
        "title": "🎓 Sertifikatlarni yuklang – mos universitetlarni ko‘ring",
        "upload": "",
        "uploaded": "{} ta rasm yuklandi.",
        "cert_type": "✅ <span style='color:#2ecc71'><b>Sertifikat turi:</b> {}</span> (<b>Aniqlik:</b> {:.1f}%), <b>Ball:</b> {}",
        "cert_not_found": "{} uchun ball aniqlanmadi.",
        "no_scores": "⚠️ Sertifikat ballari aniqlanmadi.",
        "matched_universities": "<h4 style='color:#3498db'>🎯 Sertifikatingizga mos universitetlar:</h4>",
        "no_match": "⚠️ Afsuski, mos universitetlar topilmadi.",
    }
}

# Language
lang = st.sidebar.selectbox("🌐 Language / Tilni tanlang", ["en", "uz"], index=0)
t = texts[lang]

# Title
st.markdown(f"<h2 style='color:#8e44ad'>{t['title']}</h2>", unsafe_allow_html=True)

# Upload
uploaded_files = st.file_uploader("", type=["jpg", "jpeg", "png"], accept_multiple_files=True, label_visibility="collapsed")

if uploaded_files:
    df = pd.read_csv("Uni_info_1.csv")

    st.write(t["uploaded"].format(len(uploaded_files)))

    max_width, max_height = 620, 877
    cols = st.columns(4)
    cert_scores = []

    for i, uploaded_file in enumerate(uploaded_files):
        img = Image.open(uploaded_file).convert("RGB")
        img.thumbnail((max_width, max_height))
        col = cols[i % 4]
        with col:
            st.image(img, use_container_width=True)

        cert_type, prob = classify_certificate(img)
        score = extract_score(cert_type, img)

        if score is not None:
            cert_scores.append((cert_type, score, prob))
            st.markdown(t["cert_type"].format(cert_type.upper(), prob * 100, score), unsafe_allow_html=True)
        else:
            st.error(t["cert_not_found"].format(cert_type.upper()))

    st.markdown("---")

    if not cert_scores:
        st.error(t["no_scores"])
    else:
        filtered_df = filter_universities(df, cert_scores)

        st.markdown(t["matched_universities"], unsafe_allow_html=True)

        if not filtered_df.empty:
            column_display = [
                "University name", "Country", "Bachelor/Master/PHD", "City/Country",
                "Language", "TOEFL iBT", "IELTS", "Duolingo", "SAT",
                "TOPIK_score", "TOPIK_level",
                "GPA", "Deadline", "Required Documents"
            ]
            st.data_editor(
                filtered_df[column_display].reset_index(drop=True),
                use_container_width=True,
                hide_index=True,
                num_rows="dynamic"
            )
        else:
            st.warning(t["no_match"])
else:
    st.markdown(
        "<div style='color:gray; font-size:15px;'>Drag and drop certificate images above to begin.</div>",
        unsafe_allow_html=True
    )

import streamlit as st
import gdown
import pickle

# Modelni Google Drive'dan yuklab olish
@st.cache_resource
def load_model_from_drive():
    file_id = "1LHU5bOOSVd7pw6mkQs5VoiYMhclhqxs1"  # <-- BU YERGA O'Z FAYLINGIZ ID'sini yozing
    url = f"https://colab.research.google.com/drive/1LHU5bOOSVd7pw6mkQs5VoiYMhclhqxs1?usp=sharing"
    output = "model.pkl"
    gdown.download(url, output, quiet=False)
    with open(output, "rb") as f:
        model = pickle.load(f)
    return model

model = load_model_from_drive()

# Streamlit interfeysi
st.title("🎓 Unimatch: Universitet tanlash")
user_input = st.text_input("Sertifikat natijangizni kiriting (masalan: IELTS 6.5, SAT 1150)")

if user_input:
    # Bu yerda sizning model.predict() funksiyangiz ishlaydi
    # Misol: result = model.predict([user_input])
    st.success("✅ Sizning natijalaringiz asosida universitetlar ro‘yxati tayyorlandi.")
