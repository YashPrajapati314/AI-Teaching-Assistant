import pandas as pd
import os
from typing import Literal
from classifier_performance_measurement import compute_all_metrics

def evaluate_model_performance(folder: Literal['./gpt-teacher-classifier/', './gpt-student-classifier/', './gemini-teacher-classifier/', './gemini-student-classifier/'], role: Literal['Student', 'Teacher'], single_file: str = None, to_file: str = 'gpt-teacher-classification-performance.csv'):
    ground_truth = 'final_updated_classification.json'
    
    columns = ['Model Name', 'Temperature', 'Top-k', 'Top-p', 'Hamming Loss', 'Jaccard Index', 'Precision', 'Recall', 'F1-score', 'Exact Match', 'Additional Info']
    
    data = []
    
    if single_file:
        model_results = compute_all_metrics(f'{folder}{single_file}',
                                                    role,
                                                    default_model=False)
        data.append([
                    model_results.specifications.model_name,
                    model_results.specifications.temperature,
                    model_results.specifications.top_k,
                    model_results.specifications.top_p,
                    model_results.metrics.hamming_loss,
                    model_results.metrics.jaccard_index,
                    model_results.metrics.precision,
                    model_results.metrics.recall,
                    model_results.metrics.f1_score,
                    model_results.metrics.exact_match,
                    model_results.additional_info
                ])
        
    else:
        for root, dirs, files in os.walk(folder):
            pass
        
        print()
        print(files)
        print()
        
        for file in files:
            if file != ground_truth and file.split('.')[-1] == 'json':
                if file[:7] == 'default':
                    default_model = True
                else:
                    default_model = False
                model_results = compute_all_metrics(f'{folder}{file}',
                                                    role,
                                                    default_model=default_model)


                data.append([
                    model_results.specifications.model_name,
                    model_results.specifications.temperature,
                    model_results.specifications.top_k,
                    model_results.specifications.top_p,
                    model_results.metrics.hamming_loss,
                    model_results.metrics.jaccard_index,
                    model_results.metrics.precision,
                    model_results.metrics.recall,
                    model_results.metrics.f1_score,
                    model_results.metrics.exact_match,
                    model_results.additional_info
                ])

    df = pd.DataFrame(data=data, columns=columns)

    print()
    print(df)
    print()

    df.to_csv(to_file, index=False)
    
# evaluate_model_performance(folder='./gpt-teacher-classifier/', role='Teacher')
# evaluate_model_performance(folder='./gpt-teacher-classifier/', role='Teacher', single_file='new_prompt_gpt-4o-mini_teacher_classifications--temp-0--p-0.6--k-None--.json', to_file='new-prompt-gpt-performance.csv')
# evaluate_model_performance(folder='./gpt-teacher-classifier/', role='Teacher', single_file='new_prompt_gpt-4.1-mini_teacher_classifications--temp-1.0--p-1.0--k-None--.json', to_file='new-prompt-gpt-4.1-mini-performance.csv')
# evaluate_model_performance(folder='./gpt-teacher-classifier/', role='Teacher', single_file='gpt-4o-mini_teacher_classifications--temp-0.0--p-0.6--k-None--.json', to_file='new-prompt-gpt-4.1-mini-performance.csv')
# evaluate_model_performance(folder='./gpt-teacher-classifier/', role='Teacher', to_file='gpt-teacher-classification-performance.csv')


evaluate_model_performance(folder='./gpt-student-classifier/', role='Student', to_file='gpt-student-classification-performance.csv')