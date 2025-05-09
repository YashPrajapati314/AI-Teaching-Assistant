import os
import time
import json
import string
import random
from typing import Literal, TypedDict
from dotenv import load_dotenv
import openai
from openai import OpenAI
from openai.types.beta.thread import Thread
from openai.types.beta.assistant import Assistant

class TeacherMessageClassification(TypedDict):
    message: str
    categories: list[str]

def create_new_teacher_message_classifier() -> tuple[OpenAI, Thread, Assistant]:
    load_dotenv()

    client = openai.OpenAI()

    teacher_classifier_system_prompt = """
    You are an expert model that classifies a teacher's messages into one or more of the following predefined categories. 
    - Topic Open: The teacher begins talking about a new major topic.
    - Topic Ask: The teacher asks the student what they would like to learn about, or how much detail would they like to learn it in, or in what manner exactly.
    - Importance: The teacher highlights the importance of the topic in question and makes it seem fundamental and important to understand with reasons.
    - Short Explanation: The teacher gives a short explanation of something to the student. The explanation is short because it is a part of a conversation and not because it is supposed to be superficial.
    - Detailed Explanation: The teacher gives a detailed explanation of the what, why or how of something, especially involving reasoning to understand.
    - Fact: The teacher tells a small fact or interesting piece of trivia related to the current topic to spark interest.
    - Example: The teacher gives a relevant example of something or related to something.
    - Story: The teacher narrates a story or anecdote related to the current topic when teaching about it.
    - Clarification: The teacher tries to clarify a concept or answer the student hasn't understood, has misunderstood or is confused about.
    - Answer: The teacher answers a question asked by the student.
    - Open Ask: The teacher asks an open ended question to the student. It could be for knowing a student's thought process, opinion, or level of understanding at the moment.
    - Question Ask: The teacher asks a particular question to the student, this could be to encourage them to think through by themself or to ensure they have understood the topic.
    - Answer Respond: The teacher responds to the answer provided by the student to the question they have asked or to their thoughts accordingly.
    - Connect: This includes any kind of filler messages like greetings, messages forming emotional connection, casual conversations, apologies, etc. or messages that contain those elements in significance.
    - Branch: The teacher branches to a different topic to help understand the original topic better.
    - Other: The message doesn't categorize into any of the other mentioned classes or contain a significant element that doesn't classify into any other states.

    Provide a structured output as a JSON list where each element is an object with:
    - "message": The teacher's message
    - "categories": A list of the relevant categories the message classifies into.

    The user sends pairs of student teacher messages formatted like this:-
    Student: Oh, I am not too sure what that means
    Teacher: Oh, don't worry! I will walk you through understanding it step by step...
    
    Your task is to classify only the messages of the teacher. However, you may use the student's message to understand the context better.
    """

    assistant = client.beta.assistants.create(
        name="Teacher Message Classifier",
        instructions=teacher_classifier_system_prompt,
        # tools=[{"type": "code_interpreter"}],
        model='gpt-4o-mini',
        temperature=0,
        top_p=0.6,
        response_format={
            'type': 'json_schema',
            'json_schema': {
                'name': 'message_classifications',
                'schema': {
                    'type': 'object',
                    'properties': {
                        'message': {
                            'type': 'string',
                            'description': 'The teacher\'s message'
                        },
                        'categories': {
                            'type': 'array',
                            'items': {
                                'type': 'string'
                            },
                            'description': 'A list of categories the message classifies into'
                        }
                    }
                },
                'required': ['message', 'categories']
            }
        }
    )

    thread = client.beta.threads.create()
    
    return client, thread, assistant

def classify_teacher_message(client: OpenAI, thread: Thread, assistant: Assistant, prev_student_message: str = None, teacher_message: str = None) -> list[str]:
    if prev_student_message == None:
        user_message = f'Student: \nTeacher: {teacher_message}'
    else:
        user_message = f'Student: {prev_student_message}\nTeacher: {teacher_message}'
        
    
    message = client.beta.threads.messages.create(
        thread_id=thread.id,
        role="user",
        content=user_message
    )

    run = client.beta.threads.runs.create_and_poll(
        thread_id=thread.id,
        assistant_id=assistant.id
    )

    if run.status == 'completed': 
        messages = client.beta.threads.messages.list(thread_id=thread.id)
        classification: TeacherMessageClassification = json.loads(messages.data[0].content[0].text.value)
    else:
        print(run.status)
        
    message_categories = classification['categories']

    return message_categories