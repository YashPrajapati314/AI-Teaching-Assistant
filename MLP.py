import json

with open('final_updated_classification.json') as f:
    data = json.load(f)
    
teacher_state_positions = {
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

student_state_positions = {
    'Topic Request': 16,
    'Request': 17,
    'Open Response': 18,
    'Answer': 19,
    'Correction': 20,
    'Aware': 21,
    'Unaware': 22,
    'Unclear': 23,
    'Misunderstood': 24,
    'Understood': 25,
    'Agree': 26,
    'Disagree': 27,
    'Ask Question': 28,
    'Learn Emotional': 29,
    'Pondering': 30,
    'Connect': 31,
    'Other': 32
}

no_of_teacher_states = len(teacher_state_positions.items())
no_of_student_states = len(student_state_positions.items())

n = len(data)
i = 0

input_output_pairs = []

while(i < n-2):
    curr = data[i]
    nex = data[i+1]
    nexnex = data[i+2]
    
    if curr['responder'] == 'Teacher':
        input_vector = [0 for _ in range(no_of_teacher_states + no_of_student_states)]
        output_vector = [0 for _ in range(no_of_teacher_states)]
        
        curr_states = curr['states']
        nex_states = nex['states']
        nexnex_states = nexnex['states']
        
        for state in curr_states:
            input_vector[teacher_state_positions[state]] = 1
            
        for state in nex_states:
            input_vector[student_state_positions[state]] = 1
            
        for state in nexnex_states:
            output_vector[teacher_state_positions[state]] = 1
            
        input_output_pairs.append((input_vector, output_vector))
        
    i += 1
    
