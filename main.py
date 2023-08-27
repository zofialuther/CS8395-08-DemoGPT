import streamlit as st
import langchain
from langchain import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (ChatPromptTemplate,
                                    HumanMessagePromptTemplate,
                                    SystemMessagePromptTemplate)
from langchain.document_loaders import *
from langchain.chains.summarize import load_summarize_chain
import tempfile
from langchain.docstore.document import Document

st.title('Songwriter!')

def songWriter(song_idea):
    chat = ChatOpenAI(
        model="gpt-3.5-turbo-16k",
        temperature=0.7
    )
    system_template = """You are a songwriting assistant. Your task is to generate a central 'hook' lyric and a chord progression for the verses, pre-chorus, chorus, and bridge based on the user's input."""
    system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)
    human_template = """The song idea is: '{song_idea}'. Please generate a central 'hook' lyric and a chord progression for the verses, pre-chorus, chorus, and bridge."""
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
    chat_prompt = ChatPromptTemplate.from_messages(
        [system_message_prompt, human_message_prompt]
    )

    chain = LLMChain(llm=chat, prompt=chat_prompt)
    result = chain.run(song_idea=song_idea)
    return result # returns string   

def display_hook_and_progression(hook_and_progression):
    if hook_and_progression != "":
        st.markdown(f"**Generated Hook and Chord Progression:** \n\n {hook_and_progression}")
    else:
        st.markdown("No hook and chord progression generated. Please enter a song idea.")

def lyricGenerator(hook_and_progression,user_decision):
    chat = ChatOpenAI(
        model="gpt-3.5-turbo-16k",
        temperature=0.7
    )
    system_template = """You are a song lyric generator. You generate complete lyrics based on a given 'hook' lyric and progression, but only if the user's response is affirmative."""
    system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)
    human_template = """The hook and progression is: '{hook_and_progression}'. The user's decision is: {user_decision}. If the user's decision is affirmative, please generate the complete lyrics."""
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
    chat_prompt = ChatPromptTemplate.from_messages(
        [system_message_prompt, human_message_prompt]
    )

    chain = LLMChain(llm=chat, prompt=chat_prompt)
    result = chain.run(hook_and_progression=hook_and_progression, user_decision=user_decision)
    return result # returns string   

def display_complete_lyrics(complete_lyrics):
    if complete_lyrics != "":
        st.markdown(f"**Complete Lyrics:** \n\n {complete_lyrics}")
    else:
        st.markdown("No complete lyrics generated. Please enter a song idea and confirm to complete the lyrics.")

song_idea = st.text_input('Enter mood, word, theme, or song idea')
user_decision = st.text_input('Do you want to complete the lyrics?')

if st.button('Generate Song'):
    hook_and_progression = songWriter(song_idea) if song_idea else ""
    display_hook_and_progression(hook_and_progression)

    complete_lyrics = lyricGenerator(hook_and_progression,user_decision) if hook_and_progression and user_decision else ""
    display_complete_lyrics(complete_lyrics)
