import streamlit as st

import os
os.environ["LANGCHAIN_TRACING_V2"]="true"
os.environ["LANGCHAIN_API_KEY"]=st.secrets['LANGCHAIN_API_KEY']
os.environ["LANGCHAIN_PROJECT"]="DebateAgent2"
os.environ['LANGCHAIN_ENDPOINT']="https://api.smith.langchain.com"

from openai import OpenAI
from langchain_openai import ChatOpenAI
from typing import TypedDict, Annotated, List, Dict
from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, AIMessage, ChatMessage
import random


class AgentState(TypedDict):
  agent: str
  affCase: str
  negCase: str
  judging: str
  output: str
  step: str
  topic: str

def create_llm_message(prompt:str, messages:List):
  llm_msg=[]
  llm_msg.append(SystemMessage(content=prompt))
  for msg in messages:
    llm_msg.append(HumanMessage(content=msg))
  return llm_msg

class debateAgent:
  def __init__(self):
    self.model = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key = st.secrets['OPENAI_API_KEY'])
    workflow = StateGraph(AgentState)
    workflow.add_node("Judge", self.judge)
    workflow.add_node("Aff", self.aff)
    workflow.add_node("Neg", self.neg)
    workflow.add_edge(START, "Judge")
    workflow.add_conditional_edges("Judge", self.router)
    workflow.add_conditional_edges("Aff", self.router)
    workflow.add_conditional_edges("Neg", self.router)
    self.graph = workflow.compile()

  def router(self, state: AgentState):
    print(f"inside router with {state=}")
    current_step = state['step']
    if (current_step == "AffOpen"):
      return 'Aff'
    if (current_step == "NegOpen"):
      return 'Neg'
    if (current_step == "Judgement"):
      return 'Judge'
    return END

  def judge(self, state: AgentState):
    print(f"Starting Judge with {state=}")
    current_step = state['step']
    topic = state['topic']
    # We already have topic, so need to move to Aff!!
    if (current_step == "topic"):
      next_step = "AffOpen"
      return {"output": f"Topic is {topic}", "step": next_step}
    else:
      topic=state['topic']
      aff_case=state['affCase']
      neg_case=state['negCase']
      JUDGE_PROMPT="You are an expert Judge of debates. Review the topic, AFF and NEG case below and delivery your verdict, with explanation."
      llm_messages = create_llm_message(JUDGE_PROMPT,[f"Topic: {topic}",f"Aff case: {aff_case}",f"Neg case: {neg_case}"])
      llm_response = self.model.invoke(llm_messages)
      resp=llm_response.content
      next_step = END
    return {"judging":resp, "step": next_step}
  
  def aff(self, state: AgentState):
    print(f"Starting Aff with {state=}")
    current_step = state['step']
    if (current_step == "AffOpen"):
      topic=state['topic']
      AFF_PROMPT="You are an expert debater. Create a strong AFF case for the topic below."
      llm_messages = create_llm_message(AFF_PROMPT,[topic])
      llm_response = self.model.invoke(llm_messages)
      resp=llm_response.content
      next_step = "NegOpen"
    return {"affCase":resp, "step": next_step}

  def neg(self, state: AgentState):
    print(f"Starting Neg with {state=}")
    current_step = state['step']
    if (current_step == "NegOpen"):
      topic=state['topic']
      NEG_PROMPT="You are an expert debater. Create a strong NEG case for the topic below."
      llm_messages = create_llm_message(NEG_PROMPT,[topic])
      llm_response = self.model.invoke(llm_messages)
      resp=llm_response.content
      next_step = "Judgement"
    return {"negCase":resp, "step": next_step}

st_topic=st.empty()

col1,col2,col3=st.columns(3)
col1.header("Judge")
col2.header("Aff")
col3.header("Neg")

topic = st_topic.text_input("Debate Topic")
if topic:

    app = debateAgent()
    thread_id=random.randint(1000, 9999)
    thread={"configurable":{"thread_id":thread_id}}

    for s in app.graph.stream({'step': "topic","topic":topic}, thread):
        print(f"DEBUG {s=}")
        #for k,v in s.items():
        #    if resp := v.get("output"):
        #        print(f"**** {k=}, {resp=}")
        for k,v in s.items():
            if k=='Judge':
                if v.get('step')=='AffOpen':
                   col1.write(v.get('output'))
                else:
                    col1.write(v.get("judging"))
            if k=='Aff':
                col2.write(v.get("affCase"))
            if k=='Neg':
                col3.write(v.get("negCase"))
    print("COMPLETED")