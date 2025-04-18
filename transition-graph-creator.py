import json
from typing import Literal

def create_transition_graph(filename: str = 'final_updated_classification.json', role: Literal['Student', 'Teacher', 'Both'] = 'Both'):
    with open(filename, 'r') as f:
        data = json.load(f)
    
    if role == 'Teacher':
        state_positions = {
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
    elif role == 'Student':
        state_positions = {
            
        }
    elif role == 'Both':
        state_positions = {
            
        }
    
    messages = []
    
    for object in data:
        if object != None:
            if role == 'Both':
                message_states = [state_positions[state] for state in object['states']]
                messages.append(message_states)
            elif object['responder'] == role:
                messages.append(message_states)
    
    for i in range(len(messages) - 1):
        pass