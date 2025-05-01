from openai import OpenAI
# client = OpenAI(base_url="http://localhost:5999/v1", api_key="original")
# sentence = "What's asthma?"
# messages = [{"role": "system", "content": "You are a helpful chatbot, please chat with me."},
#             {"role": "user", "content": sentence}]
# completion = client.chat.completions.create(
#   model="llama33_7b_with_openai_api",
#   messages=messages)
# print(completion.choices[0].message)


# client = OpenAI(base_url="http://172.16.34.22:4999/v1", api_key="original")
# sentence = "What's asthma?"
# messages = [{"role": "system", "content": "You are a helpful chatbot, please chat with me."},
#             {"role": "user", "content": sentence}]
# completion = client.chat.completions.create(
#   model="llama32_3b",
#   messages=messages)
# print(completion.choices[0].message)
# import requests
# import json
# num_searches = 5
# search_query = "Schwann cell embryologic origin"
# query = "http://172.16.34.21:6000/api/search?query="+search_query+"&k="+str(num_searches)
# x = requests.get(query)
# jsobj = json.loads(x.text)
# document=""
# for idx in range(len(jsobj['topk'])):
#     document = document + jsobj['topk'][idx]['text'] + "\n"
# print(document)
# import ipdb; ipdb.set_trace()
# with open("/home/htran/generation/med_preferences/raise/save/Meta-Llama-3.1-8B-Instruct_MedQA_ro1_s3_r3_textstat_t5_a1_a3_a4_a6_fm.json", "r") as f:
#     data1 = json.load(f)
# with open("/home/htran/generation/med_preferences/raise/save/Meta-Llama-3.1-8B-Instruct_MedQA_ro1_s3_r3_textstat_t5_a1_a3_a4_a7_fm.json", "r") as f:
#     data2 = json.load(f)
# with open("/home/htran/generation/med_preferences/raise/save/Meta-Llama-3.1-8B-Instruct_MedQA_ro2_s3_r3_textstat_t5_a1_a3_a4_a6_fm.json", "r") as f:
#     data3 = json.load(f)
# with open("/home/htran/generation/med_preferences/raise/save/Meta-Llama-3.1-8B-Instruct_MedQA_ro2_s3_r3_textstat_t5_a1_a3_a4_a7_fm.json", "r") as f:
#     data4 = json.load(f)

# with open("/home/htran/generation/med_preferences/raise/save/Meta-Llama-3.1-8B-Instruct_MedQA_ro2_a1_a3_a4_fm.json", "r") as f:
#     data0 = json.load(f)


import json
file_name = "/home/htran/generation/med_preferences/prime/save/Llama-3.3-70B-Instruct_MedMCQA_ro1_decompose_chat_da_v32.json"
with open(file_name, "r") as f:
    data = json.load(f)
print(file_name)
index = -1
print(data[index]["num_tested"], data[index]["accuracy"], data[index]["num_gen_correct"]/data[index]["num_tested"], data[index]["num_correct"], data[index]["num_gen_correct"])
# correct_modules = data[index]['correct_modules']
# module_acc = {key: sum(correct_modules[key])/ len(correct_modules[key]) for key in correct_modules.keys()}
# print(module_acc)
# import ipdb; ipdb.set_trace()


# print(data[-1]["num_tested"], data[-1]["accuracy"], data[-1]["num_gen_correct"]/data[-1]["num_tested"], data[-1]["num_correct"], data[-1]["num_gen_correct"])
# full_data = data[:-1]
# with open(file_name, "w") as f:
#     json.dump(full_data, f)
# import ipdb; ipdb.set_trace()
# with open("save/example.json", "w") as f:
#     json.dump(data[19], f)
# import ipdb; ipdb.set_trace()
# for idx in range(len(data)):
#     if idx == 249:
#         import ipdb; ipdb.set_trace()
import random

# List of numbers provided
# numbers = [9, 10, 11, 14, 20, 21, 23, 24]
# numbers = [ 7, 9, 17, 22, 24, 31, 35, 37, 38, 39]
# numbers = [4, 6, 9, 10, 13, 18, 20, 26, 32, 43]
# # Calculate 40% of the list size
# num_to_pick = int(len(numbers) * 0.4)
#
# # Pick random numbers
# random_numbers = random.sample(numbers, num_to_pick)
#
# print(random_numbers)



# acc, num_samples, filter_acc, filter_samples = 0.85124, 1089, 0.84008, 494
# acc, num_samples, filter_acc, filter_samples  = 0.7353, 1273, 0.77348, 362
# acc, num_samples, filter_acc, filter_samples  = 0.7353, 1273, 0.8652, 816
# acc, num_samples, filter_acc, filter_samples = 0.85124, 1089, 0.93003, 586
acc, num_samples, filter_acc, filter_samples = 0.606, 500, 0.51039, 377
# acc, num_samples, filter_acc, filter_samples = 0.606, 500, 0.72632, 290
# acc, num_samples, filter_acc, filter_samples = 0.80583, 618, 0.96269, 268
# acc, num_samples, filter_acc, filter_samples = 0.80583, 618, 85.294, 170
# acc, num_samples, filter_acc, filter_samples = 0.71336, 4183, 0.75305, 1146
# acc, num_samples, filter_acc, filter_samples = 0.71336, 4183, 0.8162, 1605
# acc, num_samples, filter_acc, filter_samples  = 0.7353, 1273, 0.77348, 362
# acc, num_samples, filter_acc, filter_samples  = 0.7353, 1273, 0.8652, 816


# actual_correct = round(num_samples * acc)
# tp = round(filter_acc * filter_samples)
# fp = filter_samples - tp
# fn = actual_correct - tp
# tn = num_samples - (tp + fp + fn)
# precision = tp / (tp + fp)
# recall = tp / (tp + fn)
# f1 = 2 * precision * recall / (precision + recall)
# acc = (tp + tn) / num_samples
# print(acc)
# print(tp, fp, fn, tn)
# print(precision, recall, f1)


# import pdb; pdb.set_trace()

# triples = [
#     ('Clostridium botulinum', 'produces', 'botulinum toxin'),
#     ('botulinum toxin', 'blocks', 'presynaptic acetylcholine release'),
#     ('botulinum toxin', 'causes', 'descending flaccid paralysis'),
#     ('botulinum toxin', 'causes', 'autonomic dysfunction'),
#     ('infant botulism', 'is caused by', 'ingestion of spores'),
#     ('infant botulism', 'presents with', 'poor suck and weak gag reflex'),
#     ('infant botulism', 'presents with', 'constipation and dry mouth'),
#     ('infant botulism', 'is diagnosed with', 'stool toxin assay'),
#     ('infant botulism', 'is treated with', 'botulism immune globulin (BIG-IV)'),
#     ('botulism', 'mimics', 'myasthenia gravis but with autonomic symptoms'),
#     ('botulism', 'does not cause', 'ascending paralysis like Guillain-Barré syndrome'),
#     ('antibiotics', 'should be avoided in', 'botulism treatment'),
#     ('honey', 'can contain', 'Clostridium botulinum spores'),
#     ('botulism', 'affects', 'neuromuscular junction'),
#     ('neuromuscular junction', 'requires', 'acetylcholine release'),
#     ('botulism', 'prevents', 'SNARE protein function'),
#     ('SNARE proteins', 'are needed for', 'neurotransmitter vesicle fusion'),
#     ('descending paralysis', 'affects', 'cranial nerves first'),
#     ('cranial nerve involvement', 'leads to', 'ptosis and weak suck'),
#     ('botulism', 'requires', 'supportive respiratory care')
# ]
# text = ""
# for trip in triples:
#     text = text + " ".join(trip) +".\n"
# print(text)

# def extract_search_queries(text):
#     queries = []
#     start_tag = "<|begin_search_query|>"
#     end_tag = "<|end_search_query|>"
#
#     # Keep searching until no more start tags are found
#     current_pos = 0
#     while current_pos < len(text):
#         # Find the next start tag
#         start_pos = text.find(start_tag, current_pos)
#         if start_pos == -1:
#             break  # No more start tags found
#
#         # Find the end tag after the start tag
#         end_pos = text.find(end_tag, start_pos)
#         if end_pos == -1:
#             break  # No corresponding end tag found
#
#         # Extract the query (text between the tags)
#         query_start = start_pos + len(start_tag)
#         query = text[query_start:end_pos].strip()
#         queries.append(query)
#
#         # Move to position after the current end tag
#         current_pos = end_pos + len(end_tag)
#
#     return queries
#
# text = "1. Review guidelines for reporting errors and complications:\n<|begin_search_query|>Reporting medical errors and near misses guidelines<|end_search_query>\n2. Consider patient autonomy and informed consent principles:\n<|begin_search_query|>patient rights disclosure of medical errors<|end_search_query>\n3. Evaluate professional responsibility:\n<|begin_search_query|>physician duty to truthfully document adverse events<|end_search_query>\n4. Assess potential consequences of each option:\nWeigh the impact of each choice on the patient's trust, safety, and well-being as well as personal and institutional liability.\n\nBased on ethical and regulatory standards, disclosing the error and documenting it in the operative report would be the appropriate course of action. This ensures transparency and allows for proper follow-up care while also protecting the institution from potential lawsuits."
#
# queries = extract_search_queries(text)
# import ipdb; ipdb.set_trace()

# from transformers import AutoProcessor, Llama4ForConditionalGeneration
# import torch
# import torch._dynamo
# torch._dynamo.config.capture_scalar_outputs = True
# torch._dynamo.config.suppress_errors = True
#
# model_id = "/data/experiment_data_gamma/hieutran/Llama-4-Scout-17B-16E-Instruct"
#
# processor = AutoProcessor.from_pretrained(model_id)
# model = Llama4ForConditionalGeneration.from_pretrained(
#     model_id,
#     attn_implementation="flash_attention_2",#"flex_attention",
#     device_map="auto",
#     torch_dtype=torch.bfloat16,
# )
#
# url1 = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/0052a70beed5bf71b92610a43a52df6d286cd5f3/diffusers/rabbit.jpg"
# url2 = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/datasets/cat_style_layout.png"
# messages = [
#     {
#         "role": "user",
#         "content": [
#             {"type": "image", "url": url1},
#             {"type": "image", "url": url2},
#             {"type": "text", "text": "Can you describe how these two images are similar, and how they differ?"},
#         ]
#     },
# ]
#
# inputs = processor.apply_chat_template(
#     messages,
#     add_generation_prompt=True,
#     tokenize=True,
#     return_dict=True,
#     return_tensors="pt",
# ).to(model.device)
#
# outputs = model.generate(
#     **inputs,
#     max_new_tokens=256,
# )
#
# response = processor.batch_decode(outputs[:, inputs["input_ids"].shape[-1]:])[0]
# print(response)
# print(outputs[0])
# import ipdb; ipdb.set_trace()


# from together import Together
#
# client = Together(
#     api_key=""
# ) # auth defaults to os.environ.get("TOGETHER_API_KEY")
#
# response = client.chat.completions.create(
#     model="meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
#     messages=[{"role": "user", "content": "What are some fun things to do in New York?"}]
# )
# print(response.choices[0].message.content)
# import i

