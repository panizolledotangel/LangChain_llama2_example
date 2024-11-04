import operator
import warnings
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, ToolMessage, AIMessage
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated

warnings.filterwarnings("always", category=UserWarning, module='transformers')

class AgentState(TypedDict):
    chat: Annotated[list[AnyMessage], operator.add]

class SortingHat:

    def __init__(self, llm, tools, sys_prompt, user_prompt):
        self.sys_prompt = sys_prompt
        self.user_prompt = user_prompt
        
        graph = StateGraph(AgentState)
        graph.add_node("llm", self.llm_call)
        graph.add_node("read_input", self.read_input)
        graph.add_node("action", self.take_action)
       
        graph.add_conditional_edges(
            "llm",
            self.exists_action,
            {True: "action", False: "read_input"}
        )
        
        graph.add_conditional_edges(
            "read_input",
            self.has_finish,
            {True: END, False: "llm"}
        )
        graph.add_edge("action", "llm")
        
        graph.set_entry_point("llm")
        
        self.graph = graph.compile()
        self.tools = {t.name: t for t in tools}
        self.llm = llm.bind_tools(tools)

    def llm_call(self, state: AgentState):    
        messages = state['chat']
        messages.insert(0, HumanMessage(content=self.user_prompt))
        messages.insert(0, SystemMessage(content=self.sys_prompt))

        promtp = ChatPromptTemplate(messages)
        chain = (ChatPromptTemplate(messages) | self.llm)
        response = chain.invoke({})
        
        print(response.content)
        return {"chat": [response]}
        
    def read_input(self, state: AgentState):
        reponse = input("User: ")
        return {"chat": [HumanMessage(content=reponse)]}

    def has_finish(self, state: AgentState):
        last_message = state["chat"][-1].content
        return last_message.lower() == "exit"

    def exists_action(self, state: AgentState):
        result = state['chat'][-1]
        return len(result.tool_calls) > 0

    def take_action(self, state: AgentState):
        tool_calls = state['chat'][-1].tool_calls
        results = []
        for t in tool_calls:
            print(f"Calling: {t}")
            if not t['name'] in self.tools:      # check for bad tool name from LLM
                print("\n ....bad tool name....")
                result = "bad tool name, retry"  # instruct LLM to retry if bad
            else:
                result = self.tools[t['name']].invoke(t['args'])
            results.append(ToolMessage(tool_call_id=t['id'], name=t['name'], content=str(result)))
        print("Back to the model!")
        return {'chat': results}

if __name__ == '__main__':

    llm = ChatOllama(
        model = "llama3.1:70b-instruct-q2_K",
        temperature = 0
    )

    sys_prompt = """
    You are Hogwarts sorting hat. You are helpfull and has a vast knowledge about the Harry Potter universe.
    """

    user_prompt = """
    You are the Hogwarts Sorting Hat, knowledgeable about the Harry Potter universe and eager to help. 
    Answer the user questions and use the search engine only when essential and only if you genuinely lack the information needed to respond accurately. 
    Before initiating a search, first determine if your own knowledge can provide a sufficient answer.
    
    Follow these rules:
    1. Only search when absolutely necessary, such as when a question cannot be answered with your existing knowledge.
    2. Avoid searching for general greetings, straightforward questions that do not require external information or questions related to the Harry Potter world.
    3. Make a search query only if you clearly understand what you need to find.
    
    Use searches sparingly, and always prioritize your own knowledge before considering external sources.
    """

    search_tool = TavilySearchResults(max_results=4)
    
    abot = SortingHat(llm, [search_tool], sys_prompt=sys_prompt, user_prompt=user_prompt)
    abot.graph.invoke({"chat": []})

