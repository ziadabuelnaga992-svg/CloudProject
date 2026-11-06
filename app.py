import streamlit as st
import pandas as pd
from fitness_chatbot import FitnessChatbot

st.set_page_config(
    page_title="Fitness AI Chatbot | NLP Project",
    page_icon="🏋️‍♂️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E88E5;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #555;
        margin-bottom: 1.5rem;
    }
    .badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 12px;
        font-size: 0.85rem;
        font-weight: 600;
        margin-right: 6px;
    }
    .badge-muscle {
        background-color: #E3F2FD;
        color: #1565C0;
    }
    .badge-category {
        background-color: #E8F5E9;
        color: #2E7D32;
    }
    .badge-score-high {
        background-color: #D4EDDA;
        color: #155724;
    }
    .badge-score-med {
        background-color: #FFF3CD;
        color: #856404;
    }
    .token-chip {
        display: inline-block;
        background-color: #ECEFF1;
        color: #37474F;
        padding: 4px 8px;
        margin: 3px;
        border-radius: 6px;
        font-family: monospace;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def get_chatbot():
    return FitnessChatbot()

bot = get_chatbot()

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "👋 Hello! I am your **Fitness & Workout AI Assistant** (NLP Project).\n\nAsk me anything about exercises, workouts, building muscle, nutrition, or weight loss!",
            "details": None
        }
    ]

if "last_query_result" not in st.session_state:
    st.session_state.last_query_result = None

with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/dumbbell.png", width=70)
    st.title("Fitness NLP Bot")
    st.caption("College NLP Course Project")
    
    st.markdown("---")
    st.subheader("⚙️ إعدادات النموذج (Model Parameters)")
    
    threshold = st.slider(
        "Confidence Threshold (حد التشابه الأدنى)",
        min_value=0.05,
        max_value=0.60,
        value=0.15,
        step=0.05
    )
    bot.confidence_threshold = threshold

    top_k = st.slider("Top-K Alternatives", min_value=1, max_value=5, value=3)

    st.markdown("---")
    st.subheader("💡 أسئلة سريعة للتجربة")
    quick_questions = [
        "How to build chest muscles?",
        "Best exercise for bigger biceps",
        "How much protein daily?",
        "How to lose belly fat?",
        "Best leg and squat workout",
        "How to get 3D shoulders?"
    ]
    for q in quick_questions:
        if st.button(q, key=f"quick_{q}"):
            st.session_state.quick_query = q

st.markdown('<div class="main-header">🏋️‍♂️ Fitness & Nutrition NLP Chatbot</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">مشروع تطبيق الـ NLP: جلب بيانات (API) ➔ معالجة نصوص (Preprocessing) ➔ متجهات TF-IDF ➔ تطابق Cosine Similarity</div>', unsafe_allow_html=True)

tab_chat, tab_inspector, tab_dataset = st.tabs([
    "💬 الشات بوت (Chat Interface)",
    "🔬 فاحص خطوات الـ NLP (Pipeline Inspector)",
    "📚 قاعدة المعرفة (Knowledge Base)"
])

with tab_chat:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"], avatar="🏋️‍♂️" if msg["role"] == "assistant" else "👤"):
            st.markdown(msg["content"])
            if msg.get("details"):
                details = msg["details"]
                if details.get("success"):
                    st.markdown(f"""
                    <div style="margin-top: 8px;">
                        <span class="badge badge-category">📁 {details.get('category')}</span>
                        <span class="badge badge-muscle">🎯 {details.get('target_muscle')}</span>
                        <span class="badge {'badge-score-high' if details.get('confidence', 0) > 40 else 'badge-score-med'}">
                            ⚡ Similarity: {details.get('confidence')}%
                        </span>
                    </div>
                    """, unsafe_allow_html=True)

    user_input = None
    if "quick_query" in st.session_state and st.session_state.quick_query:
        user_input = st.session_state.quick_query
        st.session_state.quick_query = None
    else:
        user_input = st.chat_input("Ask a workout, muscle, nutrition or exercise question...")

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input, "details": None})
        
        with st.spinner("Analyzing query via NLP pipeline..."):
            result = bot.predict(user_input, top_k=top_k)
            st.session_state.last_query_result = result
            
            st.session_state.messages.append({
                "role": "assistant",
                "content": result["answer"],
                "details": result
            })
            st.rerun()

with tab_inspector:
    st.subheader("🔬 فحص وتتبع مراحل الـ NLP Pipeline لسؤال المستخدم")
    last_res = st.session_state.last_query_result

    if not last_res or "pipeline_steps" not in last_res or not last_res["pipeline_steps"]:
        st.warning("👈 يرجى كتابة سؤال أولاً في قسم الشات لعرض مراحل معالجته بالتفصيل هنا.")
    else:
        steps = last_res["pipeline_steps"]
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### 1️⃣ المرحلة الأولى: تنظيف النص (Text Cleaning)")
            st.markdown(f"**النص الأصلي (Raw Query):** `{steps.get('step_1_raw_text')}`")
            st.markdown(f"**بعد التنظيف وإزالة الرموز (Cleaned):** `{steps.get('step_2_cleaned_text')}`")

            st.markdown("---")
            st.markdown("#### 2️⃣ المرحلة الثانية: تقسيم الكلمات (Tokenization)")
            tokens = steps.get("step_3_tokens", [])
            tokens_html = "".join([f'<span class="token-chip">{t}</span>' for t in tokens])
            st.markdown(f"**Tokens ({len(tokens)}):**<br/>{tokens_html}", unsafe_allow_html=True)

            st.markdown("---")
            st.markdown("#### 3️⃣ المرحلة الثالثة: حذف الـ Stop Words")
            filtered = steps.get("step_4_without_stopwords", [])
            filtered_html = "".join([f'<span class="token-chip" style="background-color:#C8E6C9;">{w}</span>' for w in filtered])
            st.markdown(f"**الكلمات المفتاحية المتبقية ({len(filtered)}):**<br/>{filtered_html}", unsafe_allow_html=True)

        with col2:
            st.markdown("#### 4️⃣ المرحلة الرابعة: إرجاع الجذور (Lemmatization)")
            lemmas = steps.get("step_5_lemmatized_roots", [])
            lemmas_html = "".join([f'<span class="token-chip" style="background-color:#BBDEFB;">{l}</span>' for l in lemmas])
            st.markdown(f"**جذور الكلمات اللغوية ({len(lemmas)}):**<br/>{lemmas_html}", unsafe_allow_html=True)

            st.markdown("---")
            st.markdown("#### 5️⃣ المرحلة الخامسة: أوزان متجهات الـ TF-IDF")
            keywords = steps.get("step_7_query_vector_keywords", [])
            if keywords:
                df_weights = pd.DataFrame(keywords, columns=["Term (N-gram)", "TF-IDF Weight"])
                st.dataframe(df_weights, hide_index=True)

        st.markdown("---")
        st.markdown("#### 6️⃣ المرحلة السادسة: حساب مصفوفة التشابه (Cosine Similarity Ranking)")
        top_matches = last_res.get("top_matches", [])
        if top_matches:
            match_data = []
            for i, m in enumerate(top_matches, 1):
                match_data.append({
                    "الترتيب (Rank)": f"#{i}",
                    "عنوان الموضوع / التمرين": m.get("title"),
                    "القسم (Category)": m.get("category"),
                    "العضلة المستهدفة": m.get("target_muscle"),
                    "نسبة التشابه (Cosine Score %)": f"{m.get('similarity_score')}%"
                })
            st.table(pd.DataFrame(match_data))

with tab_dataset:
    st.subheader(f"📚 استعراض قاعدة المعرفة الكاملة ({len(bot.knowledge_base)} عنصر)")
    kb_df = pd.DataFrame([
        {
            "Title": item.get("title"),
            "Category": item.get("category"),
            "Target Muscle": item.get("target_muscle"),
            "Answer / Instructions": item.get("answer")[:120] + "..."
        }
        for item in bot.knowledge_base
    ])
    st.dataframe(kb_df)
