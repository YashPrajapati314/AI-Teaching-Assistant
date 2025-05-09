from prompt_enhancer import get_next_possible_teacher_states

next_teacher_states = get_next_possible_teacher_states(prev_teacher_states = [
                                    "Connect",
                                    "Topic Ask"
                                ],
                                 student_states = [
                                    "Topic Request",
                                    "Connect"
                                 ])

next_teacher_states = get_next_possible_teacher_states(prev_teacher_states = [
                                    "Short Explanation",
                                    "Example",
                                    "Question Ask"
                                ],
                                 student_states = [
                                    "Unclear",
                                    "Misunderstood"
                                 ])

print(next_teacher_states)