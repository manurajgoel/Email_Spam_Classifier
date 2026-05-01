import streamlit as st
import pickle
import nltk
from nltk.corpus import stopwords
import string
from nltk.stem.porter import PorterStemmer

st.set_page_config(
    page_title="Spam Classifier",
    page_icon="🛡️",
    layout="centered"
)

ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return ' '.join(y)


tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))


# Header
st.title("🛡️ Spam Classifier")
st.caption("Paste any email or SMS message below to check if it's spam.")

st.divider()

input_sms = st.text_area("📩 Enter your message", placeholder="Paste your message here...", height=150)

# Button
if st.button("Analyse Message", type="primary", use_container_width=True):
    if input_sms.strip():
        with st.spinner("Analysing..."):
            transformed_sms = transform_text(input_sms)
            vector_input = tfidf.transform([transformed_sms])
            result = model.predict(vector_input)[0]

        st.divider()

        if result == 1:
            st.error("⚠️ **Spam Detected** — This message appears to be spam or unsolicited content.")
        else:
            st.success("✅ **Looks Safe** — This message does not appear to be spam.")
    else:
        st.warning("Please enter a message before clicking Analyse.")