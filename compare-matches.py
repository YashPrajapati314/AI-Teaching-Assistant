import json
from classifier_performance_measurement import exact_match, get_classification_vectors

ground_truth = 'final_updated_classification.json'

role = 'Teacher'

with open(ground_truth, 'r') as f:
        actual_classes_data = json.load(f)
        
role_messages = []

for obj in actual_classes_data:
    if obj['responder'] == role:
        role_messages.append(obj['message'])

actual_classes_vectors, predicted_classes_vectors = get_classification_vectors('./gpt-teacher-classifier/new_prompt_gpt-4o-mini_teacher_classifications--temp-0--p-0.6--k-None--.json', 'Teacher')

exact_match(actual_classes_vectors, predicted_classes_vectors, print_comparison=True, role=role, messages=role_messages)