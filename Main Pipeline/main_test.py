from student_classifier import create_new_student_message_classifier, classify_student_message
from teacher_classifier import create_new_teacher_message_classifier, classify_teacher_message
from prompt_enhancer import modified_prompt
from teacher import create_new_teacher, get_response_from_teacher

def run():
    std_msg_clsfr_client, std_msg_clsfr_thread, std_msg_clsfr_assistant = create_new_student_message_classifier()
    tch_msg_clsfr_client, tch_msg_clsfr_thread, tch_msg_clsfr_assistant = create_new_teacher_message_classifier()
    teacher_client, teacher_thread, teacher_assistant = create_new_teacher()
    
    student_message = None
    teacher_message = None
    
    while student_message != 'quit':
        
        student_message = input('You: ')
        
        if student_message == 'quit':
            break

        student_states = classify_student_message(std_msg_clsfr_client, std_msg_clsfr_thread, std_msg_clsfr_assistant, student_message=student_message)
        
        if teacher_message:
            prev_teacher_states = classify_teacher_message(tch_msg_clsfr_client, tch_msg_clsfr_thread, tch_msg_clsfr_assistant, teacher_message=teacher_message)
        else:
            prev_teacher_states = []
            
        prompt_to_send = modified_prompt(student_message=student_message, student_states=student_states, prev_teacher_states=prev_teacher_states)
        
        # print()
        # print()
        # print('Modified Prompt')
        # print(prompt_to_send)
        # print()
        # print()
        
        teacher_message = get_response_from_teacher(teacher_client, teacher_thread, teacher_assistant, user_message=prompt_to_send)

        print(f'Teacher: {teacher_message}')
        
run()