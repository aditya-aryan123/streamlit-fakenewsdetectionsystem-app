import streamlit as st
import pickle


model = pickle.load(open('model.pkl', 'rb'))
tf_idf = pickle.load(open('vectorizer.pkl', 'rb'))

st.title("Fake News Detection System")


def fakenewsdetection():
    user = st.text_area("Enter Any News Headline: ",  'Some News')
    if st.button('Predict'):
        sample = user
        data = tf_idf.transform([sample]).toarray()
        a = model.predict(data)
        st.title(a)
    else:
        print('')


fakenewsdetection()

