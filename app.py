import os
import joblib
import streamlit as st

st.set_page_config(page_title="CyberShield | Cyberbullying Detection", page_icon="🛡️", layout="wide")
MODEL_PATH = os.path.join("models", "cyberbullying_model.joblib")

st.markdown(r"""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
:root{--muted:#69758b;--purple:#7357ff;--teal:#24c8b5;--line:#e8ebf3}
html,body,[class*="css"]{font-family:'DM Sans',sans-serif}
.stApp{background:radial-gradient(circle at 88% 2%,rgba(115,87,255,.10),transparent 27rem),radial-gradient(circle at 5% 20%,rgba(36,200,181,.08),transparent 24rem),#f7f8fc;color:#172033}
.block-container{padding-top:1.6rem;padding-bottom:3rem;max-width:1180px}
h1,h2,h3{font-family:'Space Grotesk',sans-serif!important;letter-spacing:-.035em}
[data-testid="stSidebar"]{background:linear-gradient(180deg,#171b35 0%,#24234c 100%)}
[data-testid="stSidebar"] *{color:#f5f6ff!important}[data-testid="stSidebar"] .stMarkdown p{color:#c6c9e2!important}
.hero{background:linear-gradient(120deg,#202447 0%,#39306e 62%,#5142a0 100%);color:white;padding:2rem 2.2rem;border-radius:26px;box-shadow:0 18px 48px rgba(36,32,85,.18);position:relative;overflow:hidden;margin-bottom:1.25rem}
.hero:after{content:"✦";position:absolute;right:6%;top:-35px;font-size:190px;color:rgba(255,255,255,.07)}
.kicker{display:inline-block;padding:.35rem .7rem;border-radius:99px;background:rgba(36,200,181,.16);color:#8df4e6;font-size:.78rem;font-weight:700;letter-spacing:.08em;text-transform:uppercase}
.hero h1{color:white!important;font-size:clamp(2rem,4vw,3.2rem);margin:.8rem 0 .4rem}.hero p{color:#d8daf1;max-width:680px;font-size:1.03rem;line-height:1.65;margin-bottom:.25rem}
.panel{background:rgba(255,255,255,.94);border:1px solid var(--line);border-radius:20px;padding:1.25rem 1.35rem;box-shadow:0 8px 28px rgba(26,36,68,.045);margin-bottom:1rem}
.panel-title{font-family:'Space Grotesk',sans-serif;font-size:1.18rem;font-weight:700;margin-bottom:.25rem}.panel-subtitle{color:var(--muted);font-size:.91rem;margin-bottom:.9rem}
.tip{background:#eefbf8;border:1px solid #c9f1e9;color:#176e64;border-radius:14px;padding:.85rem 1rem;font-size:.9rem}
.result-card{border-radius:18px;padding:1.15rem 1.25rem;margin:.75rem 0;border:1px solid var(--line);background:white}.result-bullying{border-left:5px solid #ef6575}.result-safe{border-left:5px solid #20b99e}.result-title{font-weight:700;font-size:1.08rem}.small-muted{color:var(--muted);font-size:.88rem}
div.stButton>button{border-radius:12px;border:1px solid #e1e4ef;font-weight:700;transition:all .15s ease;min-height:2.7rem}div.stButton>button:hover{border-color:#8c7bff;color:#5d45df;transform:translateY(-1px)}div.stButton>button[kind="primary"]{background:linear-gradient(100deg,#7357ff,#5b49d8);border:none;color:white;box-shadow:0 8px 18px rgba(115,87,255,.22)}div.stButton>button[kind="primary"]:hover{color:white}
[data-testid="stTextArea"] textarea{border-radius:14px;border:1px solid #dfe3ef;background:#fcfcff;font-size:1rem;padding:1rem}[data-testid="stMetric"]{background:white;border:1px solid var(--line);border-radius:16px;padding:.8rem 1rem}hr{border-color:#e7eaf3}.footer{text-align:center;color:#8992a5;font-size:.82rem;padding-top:1rem}
</style>
<div class="hero"><span class="kicker">NLP • Machine Learning • Digital Wellbeing</span><h1>Make online spaces<br>kinder and safer.</h1><p>Check a message for language patterns that may resemble cyberbullying. The model provides a helpful signal—not a final judgment.</p></div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("## 🛡️ CyberShield")
    st.markdown("**NLP-powered message screening**")
    st.markdown("---")
    st.markdown("### How it works")
    st.markdown("1. Enter a message.\n2. TF-IDF represents its words.\n3. Logistic Regression predicts a label.\n4. Review the result with context.")
    st.markdown("---")
    st.markdown("### Model details")
    st.markdown("- Text features: TF-IDF\n- Classifier: Logistic Regression\n- Classes: bullying / not bullying")
    st.info("Student project prototype. Predictions can be wrong and should not be used alone for disciplinary decisions.")

if not os.path.exists(MODEL_PATH):
    st.error("The trained model was not found. Run `python train_model.py` first, then restart this app.")
    st.stop()
model = joblib.load(MODEL_PATH)

left, right = st.columns([1.55, 1], gap="large")
with left:
    st.markdown('<div class="panel"><div class="panel-title">✍️ Analyze a message</div><div class="panel-subtitle">Paste a chat, comment, or short post below.</div>', unsafe_allow_html=True)
    examples = {
        "Kind message": "Thank you for helping me with my project. You are very kind.",
        "Potential bullying": "You are useless and nobody wants you here.",
        "Neutral message": "Can you please explain this topic again?",
    }
    if "message_input" not in st.session_state:
        st.session_state.message_input = ""
    e1, e2, e3 = st.columns(3)
    if e1.button("😊 Kind example", use_container_width=True): st.session_state.message_input = examples["Kind message"]
    if e2.button("⚠️ Bullying example", use_container_width=True): st.session_state.message_input = examples["Potential bullying"]
    if e3.button("💬 Neutral example", use_container_width=True): st.session_state.message_input = examples["Neutral message"]
    message = st.text_area("Message text", key="message_input", placeholder="Example: Please stop making personal comments about me...", height=180, label_visibility="collapsed")
    analyze = st.button("🔎  Analyze message", type="primary", use_container_width=True)
    st.markdown('<div class="small-muted">Tip: Try different wording. Context, sarcasm, slang, and local expressions may affect results.</div></div>', unsafe_allow_html=True)
    if analyze:
        if not message.strip():
            st.warning("Please enter a message before analyzing.")
        else:
            prediction = model.predict([message])[0]
            probabilities = model.predict_proba([message])[0]
            scores = dict(zip(list(model.classes_), probabilities))
            confidence = max(probabilities) * 100
            st.markdown("### Analysis result")
            if prediction == "cyberbullying":
                st.markdown('<div class="result-card result-bullying"><div class="result-title">⚠️ Potential bullying / abusive language detected</div><p class="small-muted">The message contains patterns similar to examples labeled as cyberbullying. Review the context before deciding what to do.</p></div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="result-card result-safe"><div class="result-title">✓ No bullying detected by this model</div><p class="small-muted">The model did not find strong patterns matching its bullying examples. This does not guarantee that the message is harmless.</p></div>', unsafe_allow_html=True)
            m1, m2 = st.columns(2)
            m1.metric("Model confidence", f"{confidence:.1f}%")
            m2.metric("Characters checked", len(message))
            with st.expander("View class scores"):
                for label, score in scores.items():
                    st.write(f"**{label.replace('_', ' ').title()}** — {score * 100:.1f}%")
                    st.progress(float(score))
            st.caption("Confidence is the model's estimate, not a guarantee of correctness.")
with right:
    st.markdown('<div class="panel"><div class="panel-title">🧭 Quick guide</div><div class="panel-subtitle">Use the tool responsibly.</div>', unsafe_allow_html=True)
    st.markdown("**What it can do**\n\n- Flag wording similar to labeled examples\n- Show the model's estimated confidence\n- Help demonstrate basic NLP classification\n\n---\n\n**What it cannot reliably do**\n\n- Understand every conversation or intention\n- Reliably detect sarcasm, coded language, or all dialects\n- Decide whether a person is a bully")
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('<div class="panel"><div class="panel-title">🧪 Example messages</div><div class="panel-subtitle">Use the quick-select buttons on the left to load an example.</div><div class="tip">🌱 <b>Kind:</b> “Thanks for sharing your notes.”</div><br><div class="tip" style="background:#fff5f5;border-color:#ffdadd;color:#9b3442;">⚠️ <b>Potential bullying:</b> “Everyone should laugh at you and exclude you.”</div><br><div class="tip" style="background:#f3f5ff;border-color:#e0e4ff;color:#4b4e92;">💬 <b>Neutral:</b> “I disagree, but I respect your opinion.”</div></div>', unsafe_allow_html=True)

st.divider()
st.markdown('<div class="footer">CyberShield • College mini-project prototype • Review results with care and context</div>', unsafe_allow_html=True)
