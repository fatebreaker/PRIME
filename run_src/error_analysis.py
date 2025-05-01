import json
import requests
import re
import sys
sys.path.append(".")
from common.utils import read_txt
import pandas as pd
from openai import OpenAI
client = OpenAI(
)
parameters = {
        "model": "gpt-4.1",
        "temperature": 1.0,
        "max_completion_tokens": 1024,
        "seed": 1,
        "n": 1
    }

def extract_search_content(text):
    # Find all search queries
    query_pattern = r'<\|begin_search_query\|>(.*?)<\|end_search_query\|>'
    queries = re.findall(query_pattern, text, re.DOTALL)

    # Find all search results
    result_pattern = r'<\|begin_search_result\|>(.*?)<\|end_search_result\|>'
    results = re.findall(result_pattern, text, re.DOTALL)

    # Return both queries and results
    return [q.strip() for q in queries], [r.strip() for r in results]


def retrieve_information(search_query, corpus, top_k=3):
    document = ""
    if corpus == "textbook":
        request_query = "http://172.16.34.21:6000/api/search?query=" + search_query + "&k=" + str(top_k)
    elif corpus == "textstat":
        request_query = "http://172.16.34.21:8000/api/search?query=" + search_query + "&k=" + str(top_k)
    elif corpus == "wikipedia":
        request_query = "http://172.16.34.21:9999/api/search?query=" + search_query + "&k=" + str(top_k)
    x = requests.get(request_query)
    jsobj = json.loads(x.text)
    for idx in range(len(jsobj['topk'])):
        document = document + jsobj['topk'][idx]['text'] + "\n"
    return document


def extract_search_results(text, search_queries):
    """
    Extract search results from text based on a list of search queries.

    Args:
        text (str): The text containing search queries and their results
        search_queries (list): List of search query strings to look for

    Returns:
        list: List of extracted search results
    """
    results = []

    # Sort queries by their position in the text to process them sequentially
    query_positions = []
    for query in search_queries:
        pos = text.find(query)
        if pos != -1:
            query_positions.append((pos, query))

    # Sort by position
    query_positions.sort()

    # Extract results between queries
    for i, (pos, query) in enumerate(query_positions):
        # Find start of result (after the query)
        start_pos = pos + len(query)

        # Find end position (either next query or end of text)
        if i < len(query_positions) - 1:
            end_pos = query_positions[i + 1][0]
        else:
            end_pos = len(text)

        # Extract result and remove leading/trailing whitespace
        result = text[start_pos:end_pos].strip()

        # Remove any leading colon and whitespace
        if result.startswith(":"):
            result = result[1:].strip()

        results.append(result)

    return results


file_name = "/home/htran/generation/med_preferences/prime/save/aaa_repair.json"
with open(file_name, "r") as f:
    data = json.load(f)
# print(file_name)
# print(data[-1]["num_tested"], data[-1]["accuracy"], data[-1]["num_gen_correct"]/data[-1]["num_tested"], data[-1]["num_correct"], data[-1]["num_gen_correct"])
BEGIN_SEARCH_QUERY = "<|begin_search_query|>"
END_SEARCH_QUERY = "<|end_search_query|>"
BEGIN_SEARCH_RESULT = "<|begin_search_result|>"
END_SEARCH_RESULT = "<|end_search_result|>"

analysis_data = {
    "id": [],
    "question": [],
    "answer": [],
    "plan": [],
    "raw_search_results": [],
    "processed_search_results": [],
    "conclusion": [],
    "initial_hypothesis": [],
    "integrated_hypothesis": [],
    "query": [],
    "gpt4o_evaluation": []
}
prompt_template = read_txt("prompts/MedQA/error_analysis_prompt.txt")

for idx in range(len(data)):
    if data[idx]['modules_choice']['2']['top_choice'] == data[idx]['gold_answer']:
        continue
    # print(idx)
    question = data[idx]['question']
    answer = data[idx]['gold_solution']
    plan = data[idx]['all_solutions'][0][0][0]
    query_plan = data[idx]['all_solutions'][0][0][1]
    search_query_plan = data[idx]['all_solutions'][0][0][2]
    initial_hypothesis = data[idx]['all_solutions'][0][0][3]
    integrated_hypothesis = data[idx]['all_solutions'][0][0][4]
    index = search_query_plan.find(query_plan)
    search_query_plan = search_query_plan[:index]
    queries, results = extract_search_content(query_plan)
    results = extract_search_results(search_query_plan, queries)
    query_str = '\n'.join(queries)
    raw_results = []
    raw_text = ""
    processed_text = ""
    for jdx, query in enumerate(queries):
        raw = retrieve_information(query, "textbook")
        raw_text = raw_text + BEGIN_SEARCH_QUERY + query + END_SEARCH_QUERY + "\n"
        raw_text = raw_text + BEGIN_SEARCH_RESULT + raw + END_SEARCH_RESULT + "\n"
        processed_text = processed_text + BEGIN_SEARCH_QUERY + query + END_SEARCH_QUERY + "\n"
        processed_text = processed_text + BEGIN_SEARCH_RESULT + results[jdx] + END_SEARCH_RESULT + "\n"
    conclusion = data[idx]['all_solutions'][1][0][5]
    analysis_data['id'].append(idx)
    analysis_data['question'].append(question)
    analysis_data['answer'].append(answer)
    analysis_data['plan'].append(plan)
    analysis_data['query'].append(query_str)
    analysis_data['conclusion'].append(conclusion)
    analysis_data["raw_search_results"].append(raw_text)
    analysis_data["processed_search_results"].append(processed_text)
    analysis_data['initial_hypothesis'].append(initial_hypothesis)
    analysis_data['integrated_hypothesis'].append(integrated_hypothesis)
    prompt = prompt_template.format(question=question, answer=answer, plan=plan, query=query_str, raw=raw_text, processed=processed_text, initial=initial_hypothesis, integrated=integrated_hypothesis, conclusion=conclusion)
    messages = [{"role": "system", "content": "You are helpful AI assistant"}, {"role": "user", "content": prompt}]
    completion = client.chat.completions.create(messages=messages, **parameters)
    # import ipdb; ipdb.set_trace()
    evaluation = completion.choices[0].message.content
    analysis_data['gpt4o_evaluation'].append(evaluation)
    # import ipdb; ipdb.set_trace()
    print(len(analysis_data['id']))
    df = pd.DataFrame(analysis_data)
    output_file = "save/analysis_medqa_4.tsv"
    df.to_csv(output_file, sep='\t', index=False, encoding='utf-8')
    if len(analysis_data['id']) >= 50:
        break