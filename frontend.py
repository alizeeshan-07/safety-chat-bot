import streamlit as st

def main():
    st.title("DOXS Conversation Simulator")
    
    # Static conversation from the image
    st.write("You: Can I work under suspended load?")
    st.write("DOXS: You must use and wear all equipment as required in the work permit for working at heights. Question: what are company wages and link between them and working conditions Ansdifwfef asdfj bre sd")
    
    # Interactive input for user
    user_input = st.text_input("You: ", "Can I work under suspended load?")
    
    # (Optional) Here you can process the user input and generate a response
    # For the sake of this example, let's give a static response
    if user_input:
        st.write("DOXS: Sorry, I cannot provide that information right now.")

if __name__ == "__main__":
    main()
