import streamlit as st
from decouple import config

from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from openai import OpenAI

# Constants
PAGE_TITLE = 'Convo AI'
DEFAULT_ASSISTANT_MESSAGE = "Hello there, ask me anything...!"
DEFAULT_IMAGE_SIZE = '256x256'

# Prompt template
prompt = PromptTemplate(
    input_variables=["chat_history", "question"],
    template="""
        Your are a friendly AI Agent..
        chat_history:{chat_history}
        Human : {question}
        AI:""",
)

# Fetch OpenAI API key
try:
    openai_api_key = config("OPENAI_API_KEY")
except Exception as e:
    raise ValueError("OpenAI API key is missing or invalid. Please provide a valid key.") from e

# Initialize components
llm = ChatOpenAI(openai_api_key=openai_api_key)
openai = OpenAI(api_key=openai_api_key)
memory = ConversationBufferWindowMemory(memory_key="chat_history", k=5)
llm_chain = LLMChain(llm=llm, memory=memory, prompt=prompt)

# Set page configuration
st.set_page_config(
    page_title=PAGE_TITLE,
    layout="wide"
)
 
# Main page title
st.title(PAGE_TITLE)

# Define tabs
tab1, tab2 = st.tabs(["Chat", "Image"])

# Chat input
user_prompt = st.chat_input("Chat Input:")

# Chat tab
with tab1:
    # Initialize session state if not present
    if 'messages' not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": DEFAULT_ASSISTANT_MESSAGE}]

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # Generate AI response
    if user_prompt:
        with st.spinner("Loading..."):
            ai_response = llm_chain.predict(question=user_prompt)
        with st.chat_message('assistant'):
            st.write(ai_response)
        new_ai_message = {"role": "assistant", "content": ai_response}
        st.session_state.messages.append(new_ai_message)

# Image tab
with tab2:
    with st.form(key='form'):
        prompt_input = st.text_input(label='Enter text prompt for image generation')
        size = st.selectbox('Select size of the images', ['256x256', '512x512', '1024x1024'], index=0)
        num_images = st.selectbox('Enter number of images to be generated', (1, 2, 3, 4), index=0)
        submit_button = st.form_submit_button(label='Submit')

    # Generate images
    if submit_button and prompt_input:
        response = openai.images.generate(prompt=prompt_input, n=num_images)
        for idx, image_data in enumerate(response.data):
            image_url = image_data.url
            st.image(image_url, caption=f"Generated image: {idx+1}", use_column_width=True)

# Append user message to session state
if user_prompt:
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.write(user_prompt)