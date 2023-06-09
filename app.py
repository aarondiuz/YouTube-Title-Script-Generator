import os

import streamlit as st
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain
from langchain.memory import ConversationBufferMemory
from langchain.utilities import WikipediaAPIWrapper

# App framework
st.set_page_config(page_title="YouTube Generator", page_icon=":robot:")
st.title('🦜🔗 Youtube GPT Creator')
prompt = st.text_input('Enter the topic you want to create a title and script about')

def get_api_key():
    input_text = st.text_input(label="OpenAI API Key ",  placeholder="Ex: sk-2twmA8tfCb8un4...", key="openai_api_key_input")
    return input_text

openai_api_key = get_api_key()

# Prompt templates
title_template = PromptTemplate(
    input_variables = ['topic'],
    template = 'Write me a YouTube video title about {topic}'
)

script_template = PromptTemplate(
    input_variables = ['title', 'wikipedia_research'],
    template = 'Write me a YouTube video script (no more thatn 100 words) based on this title: {title} while leveraging this wikipedia research: {wikipedia_research}'
)

# Memory
title_memory = ConversationBufferMemory(input_key='topic', memory_key='chat_history')
script_memory = ConversationBufferMemory(input_key='title', memory_key='chat_history')

# sequential_chain = SequentialChain(chains=[title_chain, script_chain], input_variables=['topic'], output_variables=['title', 'script'])

wiki = WikipediaAPIWrapper()

# Show stuff to the screen if there's a prompt
if prompt:
    if not openai_api_key:
        st.warning('Please insert OpenAI API Key. Instructions [here](https://help.openai.com/en/articles/4936850-where-do-i-find-my-secret-api-key)', icon="⚠️")
        st.stop()
    # response = sequential_chain({'topic': prompt})
    
    # LLMs
    llm = OpenAI(temperature=0.9, openai_api_key=openai_api_key)
    # Chains
    title_chain = LLMChain(llm=llm, prompt=title_template, output_key='title', memory=title_memory)
    script_chain = LLMChain(llm=llm, prompt=script_template, output_key='script', memory=script_memory)
    
    title = title_chain.run(prompt)
    wiki_research = wiki.run(prompt)
    script = script_chain.run(title=title, wikipedia_research=wiki_research)
    response = {'title': title, 'script': script}

    st.write(response['title'])
    st.write(response['script'])

    with st.expander('Title History'):
        st.info(title_memory.buffer)
    
    with st.expander('Script History'):
        st.info(script_memory.buffer)

    with st.expander('Wikipedia Research History'):
        st.info(wiki_research)