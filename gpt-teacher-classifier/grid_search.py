from final_classifier_gpt import classify_messages_main

# p_values = [0.7, 0.8, 0.9]
# p_values = [0.4, 0.5]

# p_values = []

# for p in p_values:
#     classify_messages_main(temperature=0, top_p=p)

# optimal_p = 0.6
# new_p = 0.9

classify_messages_main(temperature=0, top_p=0.6, model_name='gpt-4o-mini', additional_info='old_prompt_third_retry')