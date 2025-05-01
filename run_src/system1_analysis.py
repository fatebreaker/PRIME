
import json
checkpoints = [
    "/home/htran/generation/med_preferences/prime/save/Meta-Llama-3.1-8B-Instruct_MedQA_ro1_da_v32.json", 118,
    "/home/htran/generation/med_preferences/prime/save/Meta-Llama-3.1-8B-Instruct_MedQA_ro2_r3_textbook_chat_da_reading_sys2_v32.json",
    "/home/htran/generation/med_preferences/prime/save/Meta-Llama-3.1-8B-Instruct_MedQA_ro2_r3_textbook_da_v64_f.json",
    "/home/htran/generation/med_preferences/prime/save/Meta-Llama-3.1-8B-Instruct_MedQA_ro3_decompose_r3_textbook_chat_da_v32_a6_a7.json",
    "/home/htran/generation/med_preferences/prime/save/Meta-Llama-3.1-8B-Instruct_MedQA_ro3_r3_textbook_chat_da_v32_a6_a7.json"
]

file_name = "/home/htran/generation/med_preferences/prime/save/Llama-3.3-70B-Instruct_MMLU_ro1_decompose_chat_da_v32.json"
with open(file_name, "r") as f:
    data = json.load(f)


sys1_correct = 0
threshold = 0.99
num_sys1 = 0
num_correct = 0
print(threshold)

for idx in range(len(data)):
    sys1_choice = data[idx]['best_choice']
    sys1_score = data[idx]['choices_info'][sys1_choice]['score']
    gold_choice = data[idx]['gold_answer']
    if sys1_score >= threshold:
        num_sys1 += 1
        if sys1_choice == gold_choice:
            sys1_correct += 1
    if sys1_choice == gold_choice:
        num_correct +=1

other_correct = num_correct - sys1_correct

print(num_correct/len(data), sys1_correct/num_sys1 ,num_sys1, len(data) - num_sys1)
print(other_correct/(len(data) - num_sys1))

