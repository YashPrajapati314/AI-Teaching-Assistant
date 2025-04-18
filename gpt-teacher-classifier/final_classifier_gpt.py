import os
import time
import json
import string
import random
import openai
from typing import Literal
from dotenv import load_dotenv


def create_model(temperature = None, top_p = None, model_name = Literal['gpt-4o-mini', 'gpt-4.1-mini']):
    
    load_dotenv()

    client = openai.OpenAI()
    
    # old system prompt
    system_prompt = """
    You are an expert AI that classifies a teacher's messages into one or more of the following predefined categories. 
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
        # name="",
        instructions=system_prompt,
        # tools=[{"type": "code_interpreter"}],
        model=model_name,
        temperature=temperature,
        top_p=top_p,
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

def classify_message(user_message: str, client: openai, thread, assistant):
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
        messages = client.beta.threads.messages.list(
            thread_id=thread.id
        )
        # print(messages)
        # print()
        # print(messages.data[0].content[0].text.value)
        classification = json.loads(messages.data[0].content[0].text.value)
    else:
        print(run.status)

    return classification

def classify_messages_one_by_one(messages, client, thread, assistant):
    message_classifications = []
    total = len(messages)
    i = 0
    while i+1 < total:
        student_teacher_message_pair = f"{messages[i]['responder']}: {messages[i]['message']}" + '\n' + f"{messages[i+1]['responder']}: {messages[i+1]['message']}"
        print(student_teacher_message_pair)
        print()
        classification = classify_message(student_teacher_message_pair, client, thread, assistant)
        print(f'Classifying messages... {(i+2)//2} of {total//2} done')
        print()
        print(classification)
        print()
        message_classifications.append(classification)
        time.sleep(1)
        i += 2
    return message_classifications

def classify_messages_main(ground_truth: str = 'final_updated_classification.json', model_name: Literal['gpt-4o-mini', 'gpt-4.1-mini'] = 'gpt-4o-mini', temperature = None, top_p = None, top_k = None, additional_info = None, custom_filename = None):
    with open(ground_truth) as f:
        data = json.load(f)

    top_k = None
    
    client, thread, assistant = create_model(temperature, top_p, model_name=model_name)

    final_classification_teacher = classify_messages_one_by_one(data, client, thread, assistant)
    
    if custom_filename:
        if os.path.exists(f'{custom_filename}.json'):
            print('File with the specified name already exists in the directory')
            random_name = ''.join(random.choices(string.ascii_letters + string.digits, k=32))
            with open(f'{random_name}.json', 'w') as f:
                json.dump(final_classification_teacher, f, indent=4)
            print(f'Dumped data into a file with random name instead: {random_name}.json')
        else:
            with open(f'{custom_filename}.json', 'w') as f:
                json.dump(final_classification_teacher, f, indent=4)
    else:
        if additional_info:
            with open(f'{additional_info}+{model_name}_teacher_classifications--temp-{temperature}--p-{top_p}--k-{top_k}--.json', 'w') as f:
                json.dump(final_classification_teacher, f, indent=4)
        else:
            with open(f'{model_name}_teacher_classifications--temp-{temperature}--p-{top_p}--k-{top_k}--.json', 'w') as f:
                json.dump(final_classification_teacher, f, indent=4)

    # with open('default_gpt-4o-mini_teacher_classifications.json', 'w') as f:
    #     json.dump(final_classification_teacher, f, indent=4)




if __name__ == '__main__':
    with open('final_updated_classification.json') as f:
        data = json.load(f)

    # print(data)
    # print(type(data))

    student_messages = []
    teacher_messages = []

    for object in data:
        if object != None:
            if(object['responder'] == 'Student'):
                student_messages.append(object['message'])
            else:
                teacher_messages.append(object['message'])
                
    client, thread, assistant = create_model(model_name='gpt-4o-mini')

    final_classification_teacher = classify_messages_one_by_one(data, client, thread, assistant)

    with open('default+gpt-4o-mini_teacher_classifications.json', 'w') as f:
        json.dump(final_classification_teacher, f, indent=4)