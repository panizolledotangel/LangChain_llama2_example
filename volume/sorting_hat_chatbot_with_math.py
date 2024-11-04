import operator
import warnings
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, ToolMessage, AIMessage
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, List

warnings.filterwarnings("always", category=UserWarning, module='transformers')


@tool
def eval_math_expression(expression: Annotated[str, "scale factor"]) -> float:
    """Evaluates a string containing math operations and return a single floating point value"""
    return eval(expression)

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
    You are Hogwarts sorting hat. You are helpfull and have a vast knowledge about the Harry Potter universe.
    """

    user_prompt = """
    You are the Hogwarts Sorting Hat, knowledgeable about the Harry Potter universe and eager to help.
    
    Answer user questions to the best of your ability, relying primarily on your knowledge of the Harry Potter world. If you encounter a question requiring a calculation or mathematical evaluation, use the math evaluation tool to perform the necessary calculations accurately.
    
    Guidelines:
    1. Only use the math evaluation tool when a question clearly requires a mathematical calculation.
    2. For general questions or inquiries based on Harry Potter lore, respond directly without using the math tool.
    3. Prioritize answering without calculations unless it’s essential for the response.
    
    Always ensure your answers are grounded in the Harry Potter universe, using the math tool sparingly and only as needed.
    """
    
    abot = SortingHat(llm, [eval_math_expression], sys_prompt=sys_prompt, user_prompt=user_prompt)
    abot.graph.invoke({"chat": []})

