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

class StudentMessageClassification(TypedDict):
    message: str
    categories: list[str]

def create_new_student_message_classifier() -> tuple[OpenAI, Thread, Assistant]:
    load_dotenv()

    client = openai.OpenAI()

    student_classifier_system_prompt = """
    You are an expert model that understands a student's state and classifies a their messages into one or more of the following predefined categories.
    - Topic Request: The student requests to learn about a particular topic, usually at the start of a conversation.
    - Request: The student requests elaboration on something or clarification of something they haven't understood.
    - Open Response: The student responds to an open ended question asked by the teacher.
    - Answer: The student answers a factual question asked by the teacher. The answer may or may not be correct.
    - Correction: The student tries to correct the teacher where they feel the teacher might be going wrong. This could also include correction of bias in the teacher's response that the student has detected.
    - Aware: The student is aware of something that the teacher has mentioned or is talking about.
    - Unaware: The student is unaware of what the teacher has mentioned or is talking about.
    - Unclear: The student has not understood something that has been explained. This could be explicitly stated by the student or implicitly understood by the teacher.
    - Misunderstood: The student has misunderstood something that has been explained.
    - Understood: The student has likely clearly understood what was explained.
    - Agree: The student agrees with what the teacher has said.
    - Disagree: The student disagrees with what the teacher has said.
    - Ask Question: The student asks a question to the teacher.
    - Learn Emotional: The emotional state of the student can be clearly seen in their response implicitly while learning. Either they are curious, distracted, surprised, happy, unhappy, or satisfied.
    - Pondering: The student is putting thought into something.
    - Connect: This includes any kind of filler messages like greetings, messages forming emotional connection, casual conversations, gratitude, apologies, etc. or messages that contain those elements in significance.
    - Other: The message doesn't categorize into any of the other mentioned classes or contain a significant element that doesn't classify into any other states.

    Provide a structured output as a JSON list where each element is an object with:
    - "message": The student's message
    - "categories": A list of the relevant categories the message classifies into.

    You will be sent pairs of teacher student messages formatted like this:-
    Teacher: Can you think of why this happens?
    Student: Is it because of the moving charges?

    Important things to consider:
    - The response to a question from the teacher is likely to be an asnwer from the student.
    - Just because a message ends with a question mark doesn't mean it is a question. Students might often end their answers with a question mark when they are uncertain while answering.

    Your task is to classify only the messages of the student. The teacher's messages are to understand the context of conversation better.
    """

    assistant = client.beta.assistants.create(
        name="Student Message Classifier",
        instructions=student_classifier_system_prompt,
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
                            'description': 'The student\'s message'
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

def classify_student_message(client: OpenAI, thread: Thread, assistant: Assistant, prev_teacher_message: str = None, student_message: str = None) -> list[str]:
    if prev_teacher_message == None:
        user_message = f'Teacher: \nStudent: {student_message}'
    else:
        user_message = f'Teacher: {prev_teacher_message}\nStudent: {student_message}'
        
    
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
        classification: StudentMessageClassification = json.loads(messages.data[0].content[0].text.value)
    else:
        print(run.status)
        
    message_categories = classification['categories']

    return message_categories