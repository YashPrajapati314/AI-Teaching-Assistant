import json
import networkx as nx
import matplotlib.pyplot as plt
from typing import Literal
import pandas as pd

def create_transition_graph(filename: str = 'final_updated_classification.json', role: Literal['Student', 'Teacher', 'Both'] = 'Both'):
    with open(filename, 'r') as f:
        data = json.load(f)

    # Define state positions based on role
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
            'Other': 16,
        }
    elif role == 'Both':
        state_positions = {
            
        }

    reversed_state_positions = {value: key for key, value in state_positions.items()}
    no_of_states = len(state_positions)
    graph = [[0 for _ in range(no_of_states)] for _ in range(no_of_states)]
    
    messages = []

    for obj in data:
        if obj:
            if role == 'Both':
                message_states = [state_positions[state] for state in obj['states']]
                messages.append(message_states)
            elif obj['responder'] == role:
                message_states = [state_positions[state] for state in obj['states']]
                messages.append(message_states)

    for m in range(len(messages) - 1):
        prev_states = messages[m]
        next_states = messages[m+1]
        for i in prev_states:
            for j in next_states:
                graph[i][j] += 1

    # --- Visualize with NetworkX ---
    G = nx.DiGraph()

    for idx, name in reversed_state_positions.items():
        G.add_node(name)

    for i in range(no_of_states):
        for j in range(no_of_states):
            if graph[i][j] > 0:
                from_state = reversed_state_positions[i]
                to_state = reversed_state_positions[j]
                G.add_edge(from_state, to_state, weight=graph[i][j])

    pos = nx.spring_layout(G, seed=20, k=0.5)  # Layout for consistent positioning
    edge_labels = nx.get_edge_attributes(G, 'weight')

    plt.figure(figsize=(200, 200))
    nx.draw(G, pos, with_labels=True, node_size=1000, node_color="lightblue", font_size=10, arrows=True)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)
    plt.title(f"{role} State Transition Graph", fontsize=14)
    # plt.tight_layout()
    # plt.show()
    
    return state_positions, reversed_state_positions, graph

def create_table(role: Literal['Student', 'Teacher', 'Both'] = 'Both'):
    state_positions, reversed_state_positions, graph = create_transition_graph(role=role)
    columns = ['Previous State']
    columns.extend(list(state_positions.keys()))
    data = []
    for i, graph_row in enumerate(graph):
        row = [reversed_state_positions[i]]
        row.extend(graph_row)
        data.append(row)
        
    df = pd.DataFrame(columns=columns, data=data)
        
    print(df)
    
    if role == 'Both':
        df.to_csv(f'Student Teacher State Transition', index=False)
    else:
        df.to_csv(f'{role} State Transition.csv', index=False)
    
create_table('Teacher')

