import streamlit as st
import predict



def main():
    # st.title("Tweet Sentiment")
    html_temp = """
                <div style="backgorund-color:tomato;padding:10px">
                <h2 style="color:white;text-align:center;"> Tweet Sentiment Finder App </h2>
                </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)
    sample_tweet = st.text_input("Tweet", "Type Here")
    result=""
    if st.button("Predict Sentiment"):
        result = predict.predict(sample_tweet)
    st.success("The Sentiment is {}".format(result))

if __name__=="__main__":
    main()
