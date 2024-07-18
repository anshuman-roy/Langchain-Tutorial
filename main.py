## Integrate with OpenAI using our API key

import os

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain, RetrievalQA
from langchain.chains.sequential import SequentialChain
from langchain.memory import ConversationBufferMemory
from langchain.schema import HumanMessage, SystemMessage, AIMessage

import streamlit as st

from dotenv import load_dotenv
from load_document import get_retriever_handle


load_dotenv()
#os.environ["OPENAI_API_KEY"] = openai_key
# streamlit framework

st.set_page_config(page_title='Class X Science Q&A Chatbot')

# Set Streamlit session state

st.title("Q&A Demo with OpenAI API")
st.header('AI Teacher')


if 'flowmessage' not in st.session_state:

    st.session_state['flowmessage'] = [
        SystemMessage(content="You are a high school teacher")
    ]

with st.sidebar:

    st.title("Content")
    with st.expander("Chapter 1"):
        st.markdown('''
            - ggjg
            - ihkhl
            - lkhkhkl
        ''')
    with st.expander("Chapter 2"):
        st.write("ijih")

# Prompt Templates
first_prompt = PromptTemplate(
    input_variables=['context', 'question'],
    template='''You are a high school teacher, given the following context and the question, generate an answer
             that can be explained to a high school student. Do not include unnecessary activities or text that is
             not relevant to the question. If the proper answer is not included in the context do not make up an 
             answer. Just say I don't know in that case.
             
             CONTEXT : {context}
             
             QUESTION : {question}
             '''
)

# OpenAI LLMs
llm = ChatOpenAI(temperature=0.7, model='gpt-4o', openai_api_key=os.getenv('OPENAI_API_KEY'))


# Memory

#person_details_memory = ConversationBufferMemory(input_key = 'question', memory_key= 'topic_details')
#milestones_memory = ConversationBufferMemory(input_key = 'topic_details', memory_key= 'topic_example')


with st.form(key='question_form'):

    input_text = st.text_input("What question you want to ask from the AI Teacher?", key='question')
    submit = st.form_submit_button('Ask AI')

    if input_text and submit:


        # llm([
        #     SystemMessage(content='You are a High School Teacher'),
        #     HumanMessage(content='Please provide some knowledge on a topic that is usually taught in a high school')
        # ])

        # chain1 = LLMChain(llm=llm, prompt=first_prompt, verbose=True, output_key='topic_details',
        #                   memory=person_details_memory)

        chain1 = RetrievalQA.from_chain_type(llm = llm,
                                             chain_type='stuff',
                                             retriever = get_retriever_handle(),
                                             input_key = "question",
                                             return_source_documents = True,
                                             chain_type_kwargs={"prompt": first_prompt}
        )

        # chain2 = LLMChain(llm=llm, prompt=second_prompt, verbose=True, output_key='topic_questions',
        #                   memory=milestones_memory)

        #st.write(chain1.run(input_text))


        # parent_chain = SequentialChain(
        #     chains=[chain1, chain2],
        #     input_variables=['topic'],
        #     output_variables=['topic_details', 'topic_questions'],
        #     verbose=True,
        # )

        st.subheader('AI says:')
        response = chain1.invoke({'question': input_text})

        st.write(response['result'])
        #st.write(response)
        #st.subheader("MCQ Questions")
        #submit2 = st.button("Generate MCQs")

        #if submit2:



        #st.write(response['topic_questions'])
#if input_text and submit:
    #if("I don't know" not in response['result']):
with st.form(key='exercise_form'):
    st.write("Do you want to see some exercises related to it?")
    a = st.radio('Pick one', ['Yes', 'No'])
    exe_btn = st.form_submit_button('Submit')
    if(exe_btn and a == 'Yes'):

        second_prompt = PromptTemplate(
            input_variables=['context'],
            template='''From the following context extract the numerical questions or the problem statements and also 
                        the solution to the problem statements. Do not make up the solution or try to solve the problem 
                        with yourself if the solution is not provided in the context. Include only the numerical questions
                        and not the simple activities.
                        
                        TEXT : {context}                     
                     '''
        )

        chain2 = RetrievalQA.from_chain_type(llm=llm,
                                             chain_type='stuff',
                                             retriever=get_retriever_handle(),
                                             input_key="exercises",
                                             return_source_documents=True,
                                             chain_type_kwargs={"prompt": second_prompt})

        exercises_response = chain2.invoke({'exercises': 'Exercises and solutions to the topic ' + input_text})
        st.write(exercises_response['result'])

        st.write(exercises_response)
