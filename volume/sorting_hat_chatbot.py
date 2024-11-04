import operator
import warnings
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, ToolMessage, AIMessage
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated

warnings.filterwarnings("always", category=UserWarning, module='transformers')

class AgentState(TypedDict):
    chat: Annotated[list[AnyMessage], operator.add]

class SortingHat:

    def __init__(self, llm, system, user):

        self.llm = llm
        self.system = system
        self.user = user
        
        graph = StateGraph(AgentState)
        graph.add_node("llm", self.llm_call)
        graph.add_node("read_input", self.read_input)
        
        graph.add_conditional_edges(
            "read_input",
            self.has_finish,
            {True: END, False: "llm"}
        )
        graph.add_edge("llm", "read_input")
        
        graph.set_entry_point("llm")
        self.graph = graph.compile()

    def llm_call(self, state: AgentState):    
        messages = state['chat']
        messages.insert(0, HumanMessage(content=self.user))
        messages.insert(0, SystemMessage(content=self.system))

        promtp = ChatPromptTemplate(messages)
        response = (ChatPromptTemplate(messages) | self.llm | StrOutputParser()).invoke({})
        
        print(response)
        return {"chat": [AIMessage(content=response)]}
        
    def read_input(self, state: AgentState):
        reponse = input("User: ")
        return {"chat": [HumanMessage(content=reponse)]}

    def has_finish(self, state: AgentState):
        last_message = state["chat"][-1].content
        return last_message.lower() == "exit"

if __name__ == '__main__':

    llm = ChatOllama(
        model = "llama3.2",
        temperature = 0.2,
        num_predict = 256
    )
    
    sh = SortingHat(llm, system="You are Hogwarts sorting hat. You are helpfull and has a vast knowledge about the Harry Potter universe", user="You are the sorting hat, introduce yourself and respond the questions asked to you")
    sh.graph.invoke({"chat": []})

