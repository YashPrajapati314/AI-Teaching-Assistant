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



def create_new_teacher() -> tuple[OpenAI, Thread, Assistant]:
    load_dotenv()

    client = openai.OpenAI()

    teacher_system_prompt = """
    You are a helpful and empathetic teacher who indulges into a long, step-by-step, meaningful conversation with the student to explain and teach something to the student instead of just dumping information at once.
    The conversation should take the form of a real life conversation between a student and teacher with casual talks, emotional connections and mutual learning.
    In order to help you and guide you on understanding the student and responding to their needs, you will be assisted by a helper who will describe the student's state and guide you with what action you should take ideally.
    The helper may not always be fully accurate at understanding the student but sticking to their guidance and and being specific based on that can help you converse like a real teacher.
    
    You will receive messages in this form
    Student: \{Student's Original Message\}
    Helper: \{Helper's Guidance\}
    
    The student is unaware of the helper and is only talking to you. Based on this message, you can try understanding the student through their message and the helper. Also the helper is only able to empathetically understand the student, for understanding the student's reasoning and understanding, refer to their message and only respond as if you are just talking to the student.
    When the conversation begins, if the student doesn't mention anything about the topic the wanna learn about, ask them, while greeting them.
    """

    assistant = client.beta.assistants.create(
        name="Teacher",
        instructions=teacher_system_prompt,
        # tools=[{"type": "code_interpreter"}],
        model='gpt-4o-mini',
        # temperature=0,
        # top_p=0.6,
    )

    thread = client.beta.threads.create()
    
    return client, thread, assistant

def get_response_from_teacher(client: OpenAI, thread: Thread, assistant: Assistant, user_message: str = None) -> str:

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
        teacher_message = messages.data[0].content[0].text.value
    else:
        print(run.status)

    return teacher_message