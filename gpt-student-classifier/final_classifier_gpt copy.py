import os
import time
import json
import string
import random
import openai
from dotenv import load_dotenv

student_classifier_system_prompt = """
You are an expert model that understands a student's state and classifies a their messages into one or more of the following predefined categories.
...

Provide a structured output as a JSON list where each element is an object with:
- "message": The student's message
- "categories": A list of the relevant categories the message classifies into.

The user sends pairs of teacher student messages formatted like this:-
Teacher: Can you think of why happens?
Student: Is it because of the moving charges?

Important things to consider:
- A question from a student is very likely to have a response from the teacher that comes under the "Answer" category
- An answer from a student is very likely to have a response from the teacher that comes under the "Answer Respond" category
- Just because a message ends with a question mark doesn't mean it is a question. Students might often end their answers with a question mark when they are uncertain while answering. Which means the teacher in the next message might classify as "Answer Respond" (response to an answer) and not "Answer" (response to a question).

Your task is to classify only the messages of the student. The teacher's messages are to understand the context of conversation better."""

def create_model(temperature = None, top_p = None):
    load_dotenv()

    client = openai.OpenAI()

    system_prompt = student_classifier_system_prompt.strip()

    assistant = client.beta.assistants.create(
        # name="",
        instructions=system_prompt,
        # tools=[{"type": "code_interpreter"}],
        model="gpt-4.1-mini",
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

def classify_messages_main(ground_truth: str = 'final_updated_classification.json', temperature = None, top_p = None, top_k = None, custom_filename = None):
    with open(ground_truth) as f:
        data = json.load(f)

    top_k = None
    
    client, thread, assistant = create_model(temperature, top_p)
        
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
        with open(f'gpt-4.1-mini_teacher_classifications--temp-{temperature}--p-{top_p}--k-{top_k}--.json', 'w') as f:
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
                
    client, thread, assistant = create_model()

    final_classification_teacher = classify_messages_one_by_one(data, client, thread, assistant)

    with open('default_gpt-4.1-mini_teacher_classifications.json', 'w') as f:
        json.dump(final_classification_teacher, f, indent=4)