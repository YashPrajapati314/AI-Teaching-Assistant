import json
from typing import Literal

class ModelSpecifications:
    def __init__(self, model_name: str, temperature: (str | float), top_k: (str | int), top_p: (str | float)):
        self.model_name = model_name
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p

class ModelMetrics:
    def __init__(self, label_cardinality: float, label_density: float, hamming_loss: float, jaccard_index: float, precision: float, recall: float, f1_score: float, exact_match: float):
        self.label_cardinality = label_cardinality
        self.label_density = label_density
        self.hamming_loss = hamming_loss
        self.jaccard_index = jaccard_index
        self.precision = precision
        self.recall = recall
        self.f1_score = f1_score
        self.exact_match = exact_match
        
class ModelResults:
    def __init__(self, specifications: ModelSpecifications, metrics: ModelMetrics, additional_info: str = None):
        self.specifications = specifications
        self.metrics = metrics
        self.additional_info = additional_info
        


def get_classification_vectors(predicted_classification_filename: str, role: Literal['Teacher', 'Student'], ground_truth: str = 'final_updated_classification.json') -> tuple[list[list[int]], list[list[int]]]:
    with open(ground_truth, 'r') as f:
        actual_classes_data = json.load(f)

    with open(predicted_classification_filename, 'r') as g:
        predicted_classes_data = json.load(g)

    actual_classes = []

    for obj in actual_classes_data:
        if obj['responder']==role:
            actual_classes.append(obj['states'])
            
    predicted_classes = []

    for obj in predicted_classes_data:
        predicted_classes.append(obj['categories'])
        
    # print(actual_classes)
    # print()
    # print(predicted_classes)

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
    else:
        state_positions = {
            
        }

    no_of_states = len(state_positions)

    n = len(actual_classes)
    m = len(predicted_classes)

    actual_classes_vectors = []
    predicted_classes_vectors = []

    for i in range(n):
        class_a = [0 for _ in range(no_of_states)]
        class_b = [0 for _ in range(no_of_states)]
        for label in actual_classes[i]:
            class_a[state_positions[label]] = 1
        for label in predicted_classes[i]:
            try:
                class_b[state_positions[label]] = 1
            except KeyError:
                print(f'Invalid prediction {label} for role {role}')
        actual_classes_vectors.append(class_a)
        predicted_classes_vectors.append(class_b)

    # print()
    # print(state_positions)
    # print(actual_classes_vectors)
    # print(predicted_classes_vectors)
    
    return (actual_classes_vectors, predicted_classes_vectors)

def accuracy(actual_classes_vectors: list[list[int]], predicted_classes_vectors: list[list[int]]):
    pass

def label_cardinality(list_of_vectors: list[list[int]]):
    n = len(list_of_vectors)
    total_sum = sum([sum(vector) for vector in list_of_vectors])
    return total_sum / n

def label_density(list_of_vectors: list[list[int]]):
    n = len(list_of_vectors)
    total_sum = sum([sum(vector)/len(vector) for vector in list_of_vectors])
    return total_sum / n

def hamming_loss(actual_classes_vectors: list[list[int]], predicted_classes_vectors: list[list[int]]):
    n = len(actual_classes_vectors)
    m = len(predicted_classes_vectors)
    if(n!=m):
        print('Length mismatch error')
    else:
        l = len(actual_classes_vectors[0])
        total_loss = []
        for i in range(n):
            xor_res = [actual_classes_vectors[i][j] ^ predicted_classes_vectors[i][j] for j in range(l)]
            total_loss.append(sum(xor_res))
        return sum(total_loss) / (n * l)
    
def jaccard_index(actual_classes_vectors: list[list[int]], predicted_classes_vectors: list[list[int]]):
    n = len(actual_classes_vectors)
    m = len(predicted_classes_vectors)
    if(n!=m):
        print('Length mismatch error')
    else:
        l = len(actual_classes_vectors[0])
        jacc_ind = []
        for i in range(n):
            intersection = [actual_classes_vectors[i][j] & predicted_classes_vectors[i][j] for j in range(l)]
            union = [actual_classes_vectors[i][j] | predicted_classes_vectors[i][j] for j in range(l)]
            jacc_ind.append(sum(intersection)/sum(union))
        return sum(jacc_ind) / len(jacc_ind)
    
def precision(actual_classes_vectors: list[list[int]], predicted_classes_vectors: list[list[int]]):
    n = len(actual_classes_vectors)
    m = len(predicted_classes_vectors)
    if(n!=m):
        print('Length mismatch error')
    else:
        l = len(actual_classes_vectors[0])
        prec = []
        for i in range(n):
            true_positives = [actual_classes_vectors[i][j] & predicted_classes_vectors[i][j] for j in range(l)]
            prec.append(sum(true_positives)/sum(predicted_classes_vectors[i]))
        return sum(prec) / len(prec)
    
def recall(actual_classes_vectors: list[list[int]], predicted_classes_vectors: list[list[int]]):
    n = len(actual_classes_vectors)
    m = len(predicted_classes_vectors)
    if(n!=m):
        print('Length mismatch error')
    else:
        l = len(actual_classes_vectors[0])
        rec = []
        for i in range(n):
            true_positives = [actual_classes_vectors[i][j] & predicted_classes_vectors[i][j] for j in range(l)]
            rec.append(sum(true_positives)/sum(actual_classes_vectors[i]))
        return sum(rec) / len(rec)
    
def F1_score(actual_classes_vectors: list[list[int]], predicted_classes_vectors: list[list[int]]):
    model_precision = precision(actual_classes_vectors, predicted_classes_vectors)
    model_recall = recall(actual_classes_vectors, predicted_classes_vectors)
    return 2 * (model_precision * model_recall) / (model_precision + model_recall)

def exact_match(actual_classes_vectors: list[list[int]], predicted_classes_vectors: list[list[int]], print_comparison: bool = False, role: Literal['Teacher', 'Student'] = None, messages: list[str] = None):
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
        
    reversed_state_dict = {value: key for key, value in state_positions.items()}
    
    n = len(actual_classes_vectors)
    exact_matches = 0
    
    if print_comparison:
        print()
        
    count_of_actual_occurrences = dict()
    count_of_predicted_occurrences = dict()
    incorrectly_assigned_state_counts = dict()
    unsassigned_state_counts = dict()
    
    for i in range(n):
        actual = {reversed_state_dict[j] for j, state in enumerate(actual_classes_vectors[i]) if state == 1}
        predicted = {reversed_state_dict[j] for j, state in enumerate(predicted_classes_vectors[i]) if state == 1}
        false_positives = predicted.difference(actual)
        false_negatives = actual.difference(predicted)
        for ac in actual:
            count_of_actual_occurrences[ac] = count_of_actual_occurrences.get(ac, 0) + 1
        for pr in predicted:
            count_of_predicted_occurrences[pr] = count_of_predicted_occurrences.get(pr, 0) + 1
        if actual_classes_vectors[i] == predicted_classes_vectors[i]:
            exact_matches += 1
            if print_comparison:
                print('EXACT MATCH!')
                print()
        else:
            for fp in false_positives:
                incorrectly_assigned_state_counts[fp] = incorrectly_assigned_state_counts.get(fp, 0) + 1
            for fn in false_negatives:
                unsassigned_state_counts[fn] = unsassigned_state_counts.get(fn, 0) + 1
        if print_comparison:
            if messages:
                print(messages[i])
                print()
            print(actual, predicted)
            # print(f'False positives: {false_positives}')
            # print(f'False negatives: {false_negatives}')
            print()
            print('---')
            print()
    
    if print_comparison:
        print()
        print('Total Actual Occurrences')
        print(count_of_actual_occurrences)
        print()
        print('Total Predicted Occurrences')
        print(count_of_predicted_occurrences)
        print()
        print('Incorrectly Assigned States')
        print(incorrectly_assigned_state_counts)
        print()
        print('Unassigned States')
        print(unsassigned_state_counts)
    
    return exact_matches / n

def compute_all_metrics(file_path: str, role: Literal['Teacher', 'Student'], ground_truth: str = 'final_updated_classification.json', default_model: bool = False) -> ModelResults:
    actual_classes_vectors, predicted_classes_vectors = get_classification_vectors(file_path, role, ground_truth)
    
    # ./gpt-teacher-classifier/gpt-4o-mini_teacher_classifications--temp-0.2--p-0.8--k-1--.json
    # default_gpt-4o-mini_teacher_classifications--temp-0.2--p-0.8--k-1--.json
    
    file_name = file_path.split('/')[-1]
    
    additional_info = None
    
    if '+' in file_name:
        plusloc = file_name.index('+')
        additional_info = file_name[:plusloc]
        file_name = file_name[plusloc+1:]
        
    
    params = file_name.split('--')
    
    model_name = params[0].split('_')[0]
        
        
    if default_model:
        pass
        model_specifications = ModelSpecifications(model_name, 'default', 'default', 'default')
    else:
        temperature = float(params[1].split('-')[1])
        top_p = float(params[2].split('-')[1])
        top_k = str(params[3].split('-')[1])
        if top_k == 'None':
            top_k = None
        else:
            top_k = int(top_k)
        model_specifications = ModelSpecifications(model_name, temperature, top_k, top_p)
    
    

    actual_label_cardinality = label_cardinality(actual_classes_vectors)
    
    model_label_cardinality = label_cardinality(predicted_classes_vectors)
    
    model_label_density = label_density(predicted_classes_vectors)
    
    model_hamming_loss = hamming_loss(actual_classes_vectors, predicted_classes_vectors)
    
    model_jaccard_index = jaccard_index(actual_classes_vectors, predicted_classes_vectors)
    
    model_precision = precision(actual_classes_vectors, predicted_classes_vectors)
    
    model_recall = recall(actual_classes_vectors, predicted_classes_vectors)
    
    model_f1_score = F1_score(actual_classes_vectors, predicted_classes_vectors)
    
    model_exact_match = exact_match(actual_classes_vectors, predicted_classes_vectors, role=role)
    
    model_metrics = ModelMetrics(model_label_cardinality, model_label_density, model_hamming_loss, model_jaccard_index, model_precision, model_recall, model_f1_score, model_exact_match)

    model_results = ModelResults(model_specifications, model_metrics, additional_info)
    
    return model_results


if __name__=='__main__':
    actual_classes_vectors, predicted_classes_vectors = get_classification_vectors('./gpt-teacher-classifier/gpt_teacher_classifications.json', 'Teacher')

    print()

    print(f'Label Cardinality of Actual Vector: {label_cardinality(actual_classes_vectors)}')
    print(f'Label Cardinality of Predicted Vector: {label_cardinality(predicted_classes_vectors)}')

    print()

    print(f'Label Density of Actual Vector: {label_density(actual_classes_vectors)}')
    print(f'Label Density of Predicted Vector: {label_density(predicted_classes_vectors)}')

    print()

    print(f'Hamming Loss: {hamming_loss(actual_classes_vectors, predicted_classes_vectors)}')

    print()

    print(f'Jaccard Index: {jaccard_index(actual_classes_vectors, predicted_classes_vectors)}')
    
    print()

    model_precision = precision(actual_classes_vectors, predicted_classes_vectors)
    print(f'Precision: {model_precision}')
    
    print()

    model_recall = recall(actual_classes_vectors, predicted_classes_vectors)
    print(f'Recall: {model_recall}')
    
    print()
    
    f1_score = F1_score(actual_classes_vectors, predicted_classes_vectors)
    print(f'F1 score: {f1_score}')
    
    print()
    
    model_exact_match = exact_match(actual_classes_vectors, predicted_classes_vectors, role='Teacher')
    print(f'Exact Match: {model_exact_match}')