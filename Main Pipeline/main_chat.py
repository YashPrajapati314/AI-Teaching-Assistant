from student_classifier import create_new_student_message_classifier, classify_student_message
from teacher_classifier import create_new_teacher_message_classifier, classify_teacher_message
from prompt_enhancer import modified_prompt
from teacher import create_new_teacher, get_response_from_teacher
import openai
from openai import OpenAI
from openai.types.beta.thread import Thread
from openai.types.beta.assistant import Assistant

class LLM:
    def __init__(self, client: OpenAI, thread: Thread, assistant: Assistant):
        self.client = client
        self.thread = thread
        self.assistant = assistant
        

def prepare_message_before_sending(student_message_classifier: LLM, teacher_message_classifier: LLM, teacher: LLM, current_student_message: str, prev_teacher_message: str | None):
    student_states = classify_student_message(student_message_classifier.client, student_message_classifier.thread, student_message_classifier.assistant, student_message=current_student_message)
    if prev_teacher_message:
        prev_teacher_states = classify_teacher_message(teacher_message_classifier.client, teacher_message_classifier.thread, teacher_message_classifier.assistant, teacher_message=prev_teacher_message)
    else:
        prev_teacher_states = []
    
            
    prompt_to_send = modified_prompt(student_message=current_student_message, student_states=student_states, prev_teacher_states=prev_teacher_states)
    
    teacher_message = get_response_from_teacher(teacher.client, teacher.thread, teacher.assistant, user_message=prompt_to_send)

    return teacher_message


class ChatSession:
    def __init__(self):
        self.student_message_classifier = LLM(*create_new_student_message_classifier())
        self.teacher_message_classifier = LLM(*create_new_teacher_message_classifier())
        self.teacher = LLM(*create_new_teacher())
        self.prev_teacher_message = None

    def send_message(self, student_message: str) -> str:
        teacher_message = prepare_message_before_sending(
            self.student_message_classifier,
            self.teacher_message_classifier,
            self.teacher,
            current_student_message=student_message,
            prev_teacher_message=self.prev_teacher_message
        )
        self.prev_teacher_message = teacher_message
        return teacher_message


if __name__ == '__main__':
    session = ChatSession()
    while True:
        m = input('You: ')
        r = session.send_message(m)
        print(f'Teacher: {r}')