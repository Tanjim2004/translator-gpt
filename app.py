import streamlit as st
from transformers import pipeline

st.set_page_config(page_title="🌐 Multilingual Translator + Cultural Context Chatbot", layout="centered")

st.title("🌐 Multilingual Translator + Cultural Context Chatbot")
st.markdown(
    """
    Enter text in any language and choose a target language.  
    The AI will translate your text and provide cultural context or explain idioms related to the translation.  
    *Note: This is for educational purposes and may not be 100% accurate.*
    """
)

@st.cache_resource(show_spinner=True)
def load_gptj():
    # Load GPT-J 6B text-generation pipeline
    return pipeline("text-generation", model="EleutherAI/gpt-j-6B", device=0)  # device=0 for GPU, -1 for CPU

gptj = load_gptj()

input_text = st.text_area("Enter text to translate:", height=120)
target_language = st.selectbox(
    "Select target language:",
    ["English", "Spanish", "French", "German", "Chinese", "Hindi", "Japanese", "Arabic"]
)

if st.button("Translate + Explain Cultural Context"):
    if input_text.strip() == "":
        st.warning("Please enter some text to translate.")
    else:
        prompt = f"""
You are a helpful translator and cultural expert.

Translate this text to {target_language}:
\"\"\"{input_text}\"\"\"

After translation, explain any cultural context, idioms, or important usage details about the translation that a language learner should know.
        
Respond in this format:

Translation:
<your translation>

Cultural Context:
<your explanation>
"""
        with st.spinner("Generating translation and cultural context..."):
            outputs = gptj(prompt, max_length=300, do_sample=True, temperature=0.7, top_p=0.9)
            generated_text = outputs[0]["generated_text"]
            # Clean output by removing prompt from generated text if repeated
            result = generated_text.replace(prompt, "").strip()

            st.subheader("Output:")
            st.markdown(result)
