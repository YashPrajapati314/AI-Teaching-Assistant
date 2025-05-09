import pickle
import numpy as np
import random
import torch
import torch.nn as nn


class StatePredictorMLP(nn.Module):
    def __init__(self, input_size, output_size):
        super(StatePredictorMLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, output_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)



teacher_states_order = {
    'Topic Open': 0,
    'Topic Ask': 1,
    'Importance': 2,
    'Short Explanation': 3,
    'Detailed Explanation': 4,
    'Fact': 5,
    'Example': 6,
    'Story': 7,
    'Clarification': 8,
    'Answer': 9,
    'Open Ask': 10,
    'Question Ask': 11,
    'Answer Respond': 12,
    'Connect': 13,
    'Branch': 14,
    'Other': 15
}

student_states_order = {
    'Topic Request': 0,
    'Request': 1,
    'Open Response': 2,
    'Answer': 3,
    'Correction': 4,
    'Aware': 5,
    'Unaware': 6,
    'Unclear': 7,
    'Misunderstood': 8,
    'Understood': 9,
    'Agree': 10,
    'Disagree': 11,
    'Ask Question': 12,
    'Learn Emotional': 13,
    'Pondering': 14,
    'Connect': 15,
    'Other': 16
}

no_of_teacher_states = len(teacher_states_order.items())
no_of_student_states = len(student_states_order.items())

input_size = no_of_teacher_states + no_of_student_states
output_size = no_of_teacher_states

model = StatePredictorMLP(input_size, output_size)
model.load_state_dict(torch.load('teacher_state_predictor.pth', map_location='cpu'))
model.eval()

teacher_states_to_prompts = {
    'Topic Open': "If you are starting a new topic, begin with a slow start and a very nice introduction. You can start with something like this...",
    'Topic Ask': "If the student hasn't mentioned it already, ask the student what they would like to learn about. If they haven't given details about how much they already know or how they would like to learn it, ask it to them. If they have already mentioned it and you have all the info needed to understand their level and begin teaching.",
    'Importance': "You can highlight the importance of the topic to make the student understand its significance and feel good learning it, in the sense of how fundamental or important it is for the development of the world today.",
    'Short Explanation': "Based on what the student has understood so far, explain the next part that would come in learning.",
    'Detailed Explanation': "Continuing in a similar flow, explain the next part in a little detail with respect to how something works or why something is the way it is (if relevant in the context).",
    'Fact': "You can tell an interesting fact about the topic the student is learning about to spark their interest. Something like an interesting story behind something or an obscure fact.",
    'Example': "If possible, give an example while explaining what you are current concept or explain that thing through an example. Examples are to make the thing easier to imagine happening in real life, and not just a reiteration of what has already been said with real life parameters. Bonus if you could explain it as an analogy with something that the student likes.",
    'Story': "Check if you can explain this concept through a story or in an anecdotal form.",
    'Clarification': "Based on the student's response, understand the reason behind the need for clarification:\n1) If it was because an important point was missed by the student, try reiterating the idea with focus on the missed point.\n2) If it was because the explanation was unclear or too heavy to digest for the student, try splitting the explanation into multiple few simpler messages each of which confirm whether the student is understanding it part by part, and clarify further where they are getting stuck.\n3) If it was just a part of teaching learning process that naturally arises, eg sometimes some explanations, when done properly, are bound to bring confusion, in such cases continue that process smoothly while clearing up their confusion or assuring them that it will get cleared up soon if it is too heavy to happen at once.",
    'Answer': "Answer the student's question, but if possible, and depending on the context, if it is something that the student could think of themselves, encourage them to think through the answer themselves by asking what do they think, etc.",
    'Open Ask': "If possible in this context, ask the student an open ended question. Ex: Asking someone what they would use for the integral notation if they were inventing calculus by themselves, how would they have approached a particular problem, or what would they like to do now.",
    'Question Ask': "If this is the right moment, you can ask a question to figure out whether the student has understood what you have explained or not, or you can ask them questions that walk them through the concept by them answering themselves in a chain of explanations.",
    'Answer Respond': "Based on the student's answer (if any), respond to them appropriately. If they are correct appreciate them and tell them why it's correct. If they are wrong, correct them but try giving them hints and ask them questions that would lead the way to the answer. Even if they are not able to answer, appreciate them for trying and learning.",
    'Connect': "",
    'Branch': "",
    'Other': ""
}

student_states_to_prompts = {
    'Topic Request': "The student apparently just requested for a topic. Appreciate them in the beginning for wanting to learn about the topic.",
    'Request': "The student might be requesting to elaborate on something.",
    'Open Response': "The student responded to an open ended question that you asked. The question may or may not have any correct answers. Based on the case, appreciate and acknowledge the student's answer and/or correct them where you can if needed.",
    'Answer': "The student apparently answered your question. Focus your response on responding to the answer and understanding the student first and then continue with an appropriate approach ahead.",
    'Correction': "The student might be trying to correct you somewhere, reflect on what they said and check if what you have said might be wrong or if it needs any correction.",
    'Aware': "The student is apparently aware of what you are trying to talk about.",
    'Unaware': "The student may not be aware of what you are talking about or what they mentioned. Try to assure them and make knowing about it a good and well paced journey for them.",
    'Unclear': "The student is unclear for one of the following reasons:\n1)The student likely must have not understood something or is not clear about what was said and needs some clarification.\n2)The student may have given a correct (or incorrect) answer but may not have completely understood the reason or meaning behind it. If this is the case, help them out by clarifying it.\n3)The student is apparently confused about something. It could be a confusion that must be naturally arising as a question through the process of learning, and it needs a clarification at the moment.",
    'Misunderstood': "The student likely must have misunderstood or not understood what was said and needs some clarification. Or the student may have given a correct (or incorrect) answer but may have misunderstood the reason or meaning behind it. If this is the case, help them out by clarifying it.",
    'Understood': "The student likely has clearly understood or is understanding what is being explained.",
    'Agree': "The student likely agrees with what you are trying to say",
    'Disagree': "The student apparently doesn't agree with what you said, try to figure out the reason behind it and understnd whether they are right, or need to be corrected, or it is just an opinionated disagreement and respond politely accordingly.",
    'Ask Question': "The student might have asked you a question or might be answering something in that manner. If it fits the scenario, appreciate them for asking the question and try to figure out a good way to lead the student to the answer, in some situations it could even be by questioning them back and encouraging them to find the answer by themselves. Try to figure out whether in this case you should answer the question yourself, or answer it partially and give a hint, or encourage the student to give it a try themselves and take an appropriate decision.",
    'Learn Emotional': "The student's feelings might be visible in their messages. Try to figure out their emotional state if any and respond accordingly while empathizing with them and introducing some emotions in your responses.",
    'Pondering': "",
    'Connect': "",
    'Other': ""
}


def encode_states(prev_teacher_states, student_states):
    teacher_vector = np.zeros(len(teacher_states_order))
    student_vector = np.zeros(len(student_states_order))

    for state in prev_teacher_states:
        teacher_vector[teacher_states_order[state]] = 1
    for state in student_states:
        student_vector[student_states_order[state]] = 1

    return np.concatenate([teacher_vector, student_vector])


def get_next_possible_teacher_states(prev_teacher_states: list[str] = [], student_states: list[str] = []) -> list[str]:
    input_vector = encode_states(prev_teacher_states, student_states)
    
    with torch.no_grad():
        input_tensor = torch.tensor(input_vector, dtype=torch.float32).unsqueeze(0)  # shape: (1, input_size)
        prediction = model(input_tensor)[0].numpy()

    # Convert index back to state names
    index_to_state = {v: k for k, v in teacher_states_order.items()}

    # If prediction is a single label
    # return [index_to_state[prediction]]

    # If multilabel output (like sigmoid vector)
    return [index_to_state[i] for i, val in enumerate(prediction) if val > 0.5]
    
    
    next_possible_teacher_states = []
    if 'Topic Request' in student_states:
        next_possible_teacher_states.extend(['Topic Ask'])
    if 'Topic Ask' in prev_teacher_states:
        next_possible_teacher_states.extend(['Topic Open', 'Importance', 'Fact'])
    if 'Short Explanation' in prev_teacher_states:
        next_possible_teacher_states.extend(['Short Explanation'])
        if random.random() < 0.4:
            next_possible_teacher_states.extend(['Example'])
        if random.random() < 0.3:
            next_possible_teacher_states.extend(['Question Ask'])
        if random.random() < 0.2:
            next_possible_teacher_states.extend(['Detailed Explanation'])
        if random.random() < 0.1:
            next_possible_teacher_states.extend(['Fact'])
        if random.random() < 0.1:
            next_possible_teacher_states.extend(['Importance'])
    if 'Ask Question' in student_states:
        next_possible_teacher_states.extend(['Answer'])
        if random.random() < 0.6:
            next_possible_teacher_states.extend(['Question Ask'])
    if 'Unclear' in student_states or 'Misunderstood' in student_states:
        next_possible_teacher_states.extend(['Clarification'])
    if 'Answer' in student_states:
        next_possible_teacher_states.extend(['Answer Respond'])
        
    return next_possible_teacher_states
        

def get_prompt_for_next_teacher_states(prev_teacher_states: list[str] = [], student_states: list[str] = []) -> str:
    next_teacher_state_prompt = ''
    next_teacher_states = get_next_possible_teacher_states(prev_teacher_states, student_states)
    next_teacher_states.sort(key=lambda x: teacher_states_order[x])
    for state in next_teacher_states:
        if teacher_states_to_prompts[state] != "":
            next_teacher_state_prompt = next_teacher_state_prompt + teacher_states_to_prompts[state] + '\n'
    
    return next_teacher_state_prompt.strip()

def get_prompt_for_student_states(prev_teacher_states: list[str] = [], student_states: list[str] = []) -> str:
    student_state_prompt = ''
    student_states.sort(key=lambda x: student_states_order[x])
    for state in student_states:
        if student_states_to_prompts[state] != "":
            student_state_prompt = student_state_prompt + student_states_to_prompts[state] + '\n'
            
    return student_state_prompt.strip()

def modified_prompt(student_message: str = '', student_states: list[str] = [], prev_teacher_states: list[str] = []) -> str:
    student_states = [state for state in student_states if state in student_states_to_prompts]
    
    prev_teacher_states = [state for state in prev_teacher_states if state in teacher_states_to_prompts]
    
    student_state_prompt = get_prompt_for_student_states(prev_teacher_states, student_states)
    
    next_teacher_state_prompt = get_prompt_for_next_teacher_states(prev_teacher_states, student_states)
    
    prompt_to_be_sent = f'''Student: {student_message}
    
    Helper: {student_state_prompt}
    {next_teacher_state_prompt}
    '''
    
    return prompt_to_be_sent