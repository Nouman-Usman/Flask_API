import streamlit as st
import main
st.title("Streamlit App for main.py")
input_value = st.text_input("Enter a value:")

if st.button("Run"):
    agent = main.RAGAgent()
    result = agent.run(input_value)  # Replace with actual function call
    st.write("Result:", result)
