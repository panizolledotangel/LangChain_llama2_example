import operator
import warnings
from enum import Enum
from langchain.output_parsers.enum import EnumOutputParser
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, ToolMessage, AIMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated


warnings.filterwarnings("always", category=UserWarning, module='transformers')

system_prompt = """
You are an intelligent and helpful assistant. Your goal is to assist the user by providing clear, accurate, and helpful answers to their questions or requests. 
Be polite, concise, and detailed, aiming to offer the best possible solution. When appropriate, explain your reasoning or suggest additional information. 
If you do not have the necessary data, respond honestly and offer to assist in other ways.
"""

def llama_prompt(human, system=system_prompt, **kwargs):
    prompt = ChatPromptTemplate([
            ("system", system),
            ("human", human)
        ],
        partial_variables=kwargs
    )
    return prompt

class AgentState(TypedDict):
    person: str
    bio: str
    last_response: str
    questions: Annotated[list[AnyMessage], operator.add]

class NeedsInput(BaseModel):
    explanation: str
    is_request: bool

class SortingHat:

    def __init__(self):

        llm = ChatOllama(
            model = "llama3.1",
            temperature = 0.2,
            num_predict = 256
        )
        
        llm_json = ChatOllama(
            model = "llama3.1",
            temperature = 0,
            num_predict = 512,
            format="json"
        )

        self.describe_chain = (
            llama_prompt("Describe the personality of {name}. If {name} is not a widely recognized public figure say it, never made things up.")
            | llm 
            | StrOutputParser()
        )

        parser = JsonOutputParser(pydantic_object=NeedsInput)
        self.needs_input_chain = (
            llama_prompt(
            """"
            Your task is to determine if this text is requesting more information.
            {text}
            
            Always follow this instructions when answering: 
            {format_instructions}
            """,
            system="You are an assistant that always responds with output in JSON format. Your task is to analyze the provided input and generate an accurate JSON response that aligns with the user’s query.",
            format_instructions="""
            The output should be formatted as a JSON instance that conforms to the JSON schema below.
            
            As an example, for the schema {"properties": {"foo": {"title": "Foo", "description": "a list of strings", "type": "array", "items": {"type": "string"}}}, "required": ["foo"]}
            the object {"foo": ["bar", "baz"]} is a well-formatted instance of the schema. The object {"properties": {"foo": ["bar", "baz"]}} is not well-formatted.
            
            Here is the output schema:
            ```
            {"explanation": "string", "is_request": "boolean"}
            ```
            """
            )  
            | llm_json
            | parser
        )
        
        self.sorting_chain = (
            llama_prompt(
                """
                Act as the sorting hat and tell to what Hogwarts house will this person go. Hogwards houses have the next traits:
                Gryffindor: Brave, daring, chivalrous, and bold. They value courage and are often seen as leaders willing to stand up for what’s right.
                Hufflepuff: Loyal, hardworking, fair, and patient. Hufflepuffs prize dedication and kindness, with a strong sense of justice and inclusivity.
                Ravenclaw: Intelligent, curious, creative, and wise. Ravenclaws value learning, wit, and wisdom, always seeking knowledge and understanding.
                Slytherin: Ambitious, cunning, resourceful, and determined. Slytherins are driven, often setting high goals and valuing loyalty within their group.
                Here is the persons personality:
                "{bio}"
                Be brief.
                """,
                system="""
                You are an expert on the Harry Potter universe. Use your bast knowledge of the Harry Potter universe to perform the tasks the user demands. 
                If you don't have the necessary information to perform a task, indicate so, NEVER MADE THINGS UP.
                """
            )
            | llm 
            | StrOutputParser()
        )

        self.update_bio_chain = (
            llama_prompt(
                """
                Your task is to update someones biography. Here is the information so far:
                {actual_bio}
                Here are some questions about him:
                "{questions}"
                Here are the answers:
                "{answers}"
                Never ask new questions, just update the personality traits with the new information.
                """
            )
            | llm 
            | StrOutputParser()
        )

        self.try_sorting_chain = (
            llama_prompt(
                """
                Determine if the information provided about {name} is enough to confidently sort them into a Hogwarts House. 
                Sort them if possible based on their personality traits. 
                If you genuinely lack sufficient information to make an informed decision, ask a single, specific question about {name}'s personality traits to clarify.
                
                Be concise, and avoid asking questions if the current traits allow a reasonable sorting decision.
                Personality traits collected for {name} so far:
                ''' {bio} '''
                """,
                system="""
                Act as the Hogwarts Sorting Hat, using your deep knowledge of the Harry Potter universe to perform tasks as instructed. 
                If you cannot perform a task with the current information, state this clearly but never invent details. Follow user instructions precisely.
                """
            )
            | llm 
            | StrOutputParser()
        )
        
        graph = StateGraph(AgentState)
        graph.add_node("describe_person", self.describe_person)
        graph.add_node("try_sorting", self.try_sorting)
        graph.add_node("get_user_input", self.get_user_input)
        graph.add_node("refine_bio", self.refine_bio)
        graph.add_node("sorting", self.sorting)

        graph.add_edge("describe_person", "try_sorting")
        graph.add_conditional_edges(
            "try_sorting",
            self.needs_input,
            {True: "get_user_input", False: "sorting"}
        )
        graph.add_edge("get_user_input", "refine_bio")
        graph.add_edge("refine_bio", "try_sorting")
        graph.add_edge("sorting", END)
        
        graph.set_entry_point("describe_person")
        self.graph = graph.compile()

    def needs_input(self, state: AgentState):    
        response = self.needs_input_chain.invoke({"text": state['questions'][-1]})
        return response["is_request"]
        
    def describe_person(self, state: AgentState):
        bio = self.describe_chain.invoke({"name": state['person']})
        return {"bio": bio}

    def get_user_input(self, state: AgentState):
        print(state["questions"][-1].content)
        user_input = input("User: ")
        return {"last_response": user_input}

    def refine_bio(self, state: AgentState):
        new_bio = self.update_bio_chain.invoke({"actual_bio": state['bio'], "questions": state["questions"][-1].content, "answers": state['last_response']})
        print("======= NEW BIO =================")
        print(new_bio)
        print("==================================")
        return {"bio": new_bio}

    def try_sorting(self, state: AgentState):
        chain_input = {"bio": state['bio'], "name": state["person"]}

        if len(state["questions"]) < 4:
            response = self.try_sorting_chain.invoke(chain_input)
        else:
            response = "I have no more questions"

        return {"questions": [AIMessage(response)]}
            
    
    def sorting(self, state: AgentState):
        response = self.sorting_chain.invoke({"bio": state['bio']})
        return {"last_response": response}

if __name__ == '__main__':
    print("Who are you ?")
    person = input("User: ")
    sh = SortingHat()
    result = sh.graph.invoke({"person": person})
    print(result['last_response'])

