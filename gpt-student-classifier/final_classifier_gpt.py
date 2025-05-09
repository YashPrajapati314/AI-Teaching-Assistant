import os
import time
import json
import string
import random
import openai
from typing import Literal
from dotenv import load_dotenv

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

The user sends pairs of teacher student messages formatted like this:-
Teacher: Can you think of why this happens?
Student: Is it because of the moving charges?

Important things to consider:
- The response to a question from the teacher is likely to be an asnwer from the student.
- Just because a message ends with a question mark doesn't mean it is a question. Students might often end their answers with a question mark when they are uncertain while answering. Which means it belongs to the "Answer" category and not "Question Ask".

Your task is to classify only the messages of the student. The teacher's messages are to understand the context of conversation better."""

def create_model(temperature = None, top_p = None, model_name = Literal['gpt-4o-mini', 'gpt-4.1-mini']):
    load_dotenv()

    client = openai.OpenAI()

    system_prompt = student_classifier_system_prompt.strip()

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
    i = -1
    while i+1 < total:
        if i == -1:
            pass
            teacher_student_message_pair = f"Teacher:" + '\n' + f"{messages[i+1]['responder']}: {messages[i+1]['message']}"
        else:
            teacher_student_message_pair = f"{messages[i]['responder']}: {messages[i]['message']}" + '\n' + f"{messages[i+1]['responder']}: {messages[i+1]['message']}"
        print(teacher_student_message_pair)
        print()
        classification = classify_message(teacher_student_message_pair, client, thread, assistant)
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
            with open(f'{additional_info}+{model_name}_student_classifications--temp-{temperature}--p-{top_p}--k-{top_k}--.json', 'w') as f:
                json.dump(final_classification_teacher, f, indent=4)
        else:
            with open(f'{model_name}_student_classifications--temp-{temperature}--p-{top_p}--k-{top_k}--.json', 'w') as f:
                json.dump(final_classification_teacher, f, indent=4)

    # with open('default_gpt-4o-mini_student_classifications.json', 'w') as f:
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

    with open('default+gpt-4o-mini_student_classifications.json', 'w') as f:
        json.dump(final_classification_teacher, f, indent=4)