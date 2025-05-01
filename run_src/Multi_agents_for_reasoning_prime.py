# Licensed under the MIT license.

import sys
import requests

sys.path.append(".")
import re
import numpy as np, os, random, json, math, wandb
from tqdm import trange
from typing import List, Dict, Tuple
from copy import deepcopy
import math
try:
    from rapidfuzz import fuzz, process
except:
    pass

from models.IO_System import IO_System
from common.utils import read_txt, read_json
from eval_src.Evaluator import Evaluator, GSM8KEvaluator
# from MCTS_backbone import MCTS_Searcher, MCTS_Node
from Multi_Agent_backbone import Multi_Agents_Searcher, Multi_Agents_Node
from run_src.rstar_utils import (
    Node_Type,
    GeneratorError,
    reach_terminal_subquestion,
    reach_terminal_ost_step,
    concat_subqs_and_subas,
    concat_queries_and_docs,
    concat_ost_steps,
    concat_subqs_subas_as_ost_steps,
    make_hint,
    make_response_prefix,
    split_user_question,
    print_tree_from_root,
    find_valid_solution_nodes,
    find_best_solution,
    stochastic_find_best_solution,
)
from prompts import get_task_instruction_openqa, get_documents_to_reasonchain_instruction, get_task_instruction_medqa
BEGIN_SEARCH_QUERY = "<|begin_search_query|>"
END_SEARCH_QUERY = "<|end_search_query|>"
BEGIN_SEARCH_RESULT = "<|begin_search_result|>"
END_SEARCH_RESULT = "<|end_search_result|>"
prompt_with_options = "{question} A: {A}, B: {B}, C:{C}, D: {D}"


def verbose_print(s: str, verbose: bool):
    if verbose:
        print(s)


class Generator:
    """Generator generates children nodes"""

    def __init__(self, args, tokenizer, model, evaluator: Evaluator) -> None:
        self.io = IO_System(args, tokenizer, model)
        self.evaluator = evaluator

        self.num_subquestions = args.num_subquestions
        self.num_queries = args.num_queries
        self.num_a1_steps = args.num_a1_steps
        self.num_votes = args.num_votes
        self.max_tokens = args.max_tokens
        self.search_query_weight = args.search_query_weight
        self.enable_potential_score = args.enable_potential_score
        self.tokenizer = tokenizer
        self.max_depth_allowed = args.max_depth_allowed

        self.num_last_votes = args.num_last_votes
        self.num_retrieval = args.num_retrieval
        self.combine_distributions = args.combine_distributions
        self.enable_majority = args.enable_majority
        self.majority_threshold = args.majority_threshold
        self.retrieval_threshold = args.retrieval_threshold
        self.enable_chat_template = args.enable_chat_template
        self.enable_answer_revision = args.enable_answer_revision
        self.enable_self_reward = args.enable_self_reward
        self.enable_reading_documents = args.enable_reading_documents
        self.enable_debate_solution = args.enable_debate_solution

        with open(args.decompose_template_path, "r") as f:
            decompose_template = json.load(f)
            self.question_index = decompose_template["index"]

        self.decompose_prompt = read_txt(args.decompose_prompt_path)
        self.fewshot_cot_prompt = read_txt(args.fewshot_cot_prompt_path)
        self.fewshot_cot_config = read_json(args.fewshot_cot_config_path)

        self.fewshot_cot_rag_prompt = read_txt(args.fewshot_cot_rag_prompt_path)
        self.fewshot_cot_rag_config = read_json(args.fewshot_cot_rag_config_path)

        self.fewshot_self_reward_prompt = read_txt(args.fewshot_self_reward_prompt_path)
        self.fewshot_self_reward_config = read_json(args.fewshot_self_reward_config_path)

        self.decompose_query_prompt = read_txt(args.decompose_query_prompt_path)
        self.decompose_query_config = read_json(args.decompose_query_template_path)

        self.regen_module_adaptation_config = read_json(args.regen_module_adaptation_config_path)
        self.regen_module_adaptation_prompt = read_txt(args.regen_module_adaptation_prompt_path)

        self.regen_reasoning_structure_config = read_json(args.regen_reasoning_structure_config_path)
        self.regen_reasoning_structure_prompt = read_txt(args.regen_reasoning_structure_prompt_path)

        self.regen_reasoning_solution_config = read_json(args.regen_reasoning_solution_config_path)
        self.regen_reasoning_solution_prompt = read_txt(args.regen_reasoning_solution_prompt_path)

        self.regen_query_generation_config = read_json(args.regen_query_generation_config_path)
        self.regen_query_generation_prompt =  read_txt(args.regen_query_generation_prompt_path)

        self.regen_solution_revision_config = read_json(args.regen_solution_revision_config_path)
        self.regen_solution_revision_prompt = read_txt(args.regen_solution_revision_prompt_path)
        self.regen_reasoning_thought_prompt = read_txt(args.regen_reasoning_thought_prompt_path)
        self.regen_affirmative_prompt = read_txt(args.regen_affirmative_prompt_path)
        self.regen_negative_prompt = read_txt(args.regen_negative_prompt_path)
        self.regen_judge_prompt = read_txt(args.regen_judge_prompt_path)

        self.regen_initial_hypothesis_config = read_json(args.regen_initial_hypothesis_config_path)
        self.regen_initial_hypothesis_prompt = read_txt(args.regen_initial_hypothesis_prompt_path)

        self.regen_integrate_hypothesis_config = read_json(args.regen_integrate_hypothesis_config_path)
        self.regen_integrate_hypothesis_prompt = read_txt(args.regen_integrate_hypothesis_prompt_path)

        self.regen_decision_solution_config = read_json(args.regen_decision_solution_config_path)
        self.regen_decision_solution_prompt = read_txt(args.regen_decision_solution_prompt_path)


        self.retrieval_corpus = args.retrieval_corpus
        self.num_child_create = args.num_child_create
        self.dataset_name = args.dataset_name

        if not args.disable_a1:  # A1: Propose an one-step thought.
            self.fewshot_ost_prompt = read_txt(args.fewshot_ost_prompt_path)
            self.fewshot_ost_config = read_json(args.fewshot_ost_config_path)

        if not args.disable_a5:  # A5: Rephrase the question/sub-question.
            self.rephrasing_prompt_template = read_txt(args.rephrasing_prompt_template_path)
            self.decompose_prompt_rephrased = read_txt(args.decompose_prompt_rephrased_path)
            self.fewshot_cot_prompt_rephrased = read_txt(args.fewshot_cot_prompt_rephrased_path)
            self.fewshot_ost_prompt_rephrased = read_txt(args.fewshot_ost_prompt_rephrased_path)

    def _extract_from_cache(self, subquestion_list: List[str]):
        high_score_questions = []
        selected_answers = []
        values = []
        low_score_questions = []
        low_score_values = []
        low_score_answers_list = []
        unmatched_questions = []

        for subquestion in subquestion_list:
            best_match = process.extractOne(subquestion, self.reasoning_cache.keys(), scorer=fuzz.ratio)

            if best_match:
                best_question, best_score = best_match[0], best_match[1]
                similarity = best_score / 100
                cache_entry = self.reasoning_cache[best_question]
                score = cache_entry["score"]
                if similarity == 1:
                    if score >= 0.9:
                        high_score_questions.append(best_question)
                        selected_answers.append(cache_entry["selected_answer"])
                        values.append(score)
                    else:
                        low_score_questions.append(best_question)
                        low_score_values.append(score)
                        low_score_answers_list.append(cache_entry["answer_list"])
                else:
                    unmatched_questions.append(subquestion)
            else:
                unmatched_questions.append(subquestion)

        return {
            "high_score_questions": high_score_questions,
            "selected_answers": selected_answers,  # most likely answer corresponding to each subquestion
            "values": values,
            "low_score_questions": low_score_questions,
            "low_score_values": low_score_values,
            "low_score_answers_list": low_score_answers_list,
            "unmatched_questions": unmatched_questions,
        }

    def _get_most_likely_answer_with_options(self, io_output_list: List[str], options: dict) -> Tuple[str, float]:
        assert len(io_output_list) > 0

        # if len(io_output_list) == 1:
        #     most_confident_answer_full_completions = io_output_list
        #     confidences = [1]
        # else:
        most_confident_choices, most_confident_answer_full_completions, _, confidences = self.evaluator.find_most_confident_answer_with_options(
                io_output_list, options
        )
            # assert confidence > 0

        return most_confident_choices, most_confident_answer_full_completions, confidences

    def _get_most_likely_answer(self, io_output_list: List[str]) -> Tuple[str, float]:
        assert len(io_output_list) > 0

        if len(io_output_list) == 1:
            most_confident_answer_full_completions = io_output_list
            confidences = [1]
        else:
            most_confident_answers, most_confident_answer_full_completions, _, confidences = self.evaluator.find_most_confident_answer(
                io_output_list
            )
            # assert confidence > 0

        return most_confident_answers, most_confident_answer_full_completions, confidences

    def _fewshot_cot_answer_question(self, question):
        fewshot_cot_prompt = self.fewshot_cot_prompt
        io_input = self.fewshot_cot_config["prompt_template"].format(examples=fewshot_cot_prompt, instruction=question)
        io_output_list = self.io.generate(
            io_input,
            num_return=self.num_last_votes,
            max_tokens=self.max_tokens,
            stop_tokens=self.fewshot_cot_config["stop_tokens"],
        )
        cleaned_io_output_list = [io_output.strip() for io_output in io_output_list]
        # import ipdb; ipdb.set_trace()
        return io_input, cleaned_io_output_list

    def _fewshot_cot_rag_answer_question(self, question: str, documents:str, num_return: int):
        fewshot_cot_rag_prompt = self.fewshot_cot_rag_prompt
        io_input = self.fewshot_cot_rag_config["prompt_template"].format(examples=fewshot_cot_rag_prompt, question=question, documents=documents)
        # import ipdb; ipdb.set_trace()
        io_output_list = self.io.generate(
            io_input,
            num_return=num_return,
            max_tokens=self.max_tokens,
            stop_tokens=self.fewshot_cot_rag_config["stop_tokens"],
        )
        cleaned_io_output_list = [io_output.strip() for io_output in io_output_list]  #! cleaning
        return io_input, cleaned_io_output_list

    def _generate_multiple_queries_for_user_question(self, user_question):
        decompose_query_prompt = self.decompose_query_prompt
        existing_queries_and_docs, next_query_id = "", 1
        query_list = []
        doc_list = []
        score_list = []
        document_list = []
        num_queries = 0
        for kdx in range(1):
            # ! generate queries
            io_input = (
                    decompose_query_prompt
                    + "\n\n"
                    + f"Question {self.question_index}: {user_question}"
                    + "\n"
                    + existing_queries_and_docs
                    + f"Query {self.question_index}.{next_query_id}:"
            )
            io_output_list = self.io.generate(
                io_input,
                max_tokens=128,
                num_return=self.num_queries,#32,
                stop_tokens=[
                    "\n",
                    "\n\n",
                    "Document",
                    "Document ",
                    f"Document {self.question_index}.{next_query_id}",
                    f"Document {self.question_index}.{next_query_id}:",
                    f"Document {self.question_index}.{next_query_id}: ",
                ],
            )

            gen_queries = [o.strip() for o in io_output_list]
            for query in gen_queries:
                if self.retrieval_corpus == "textbook":
                    request_query = "http://172.16.34.21:6000/api/search?query=" + query + "&k=" + str(self.num_retrieval)
                elif self.retrieval_corpus == "textstat":
                    request_query = "http://172.16.34.21:8000/api/search?query=" + query + "&k=" + str(self.num_retrieval)
                elif self.retrieval_corpus == "textbooks_hipporag":
                    request_query = "http://172.16.34.21:7000/api/search?query=" + query + "&k=" + str(self.num_retrieval)
                x = requests.get(request_query)
                jsobj = json.loads(x.text)
                documents = []
                scores = []
                relevant_doc = ""
                for jdx in range(len(jsobj['topk'])):
                    if 'prob' in jsobj['topk'][jdx].keys():
                        score = jsobj['topk'][jdx]['prob']
                    else:
                        score = jsobj['topk'][jdx]['score']
                    if score >= self.retrieval_threshold:
                        relevant_doc = relevant_doc + jsobj['topk'][jdx]['text'] + "\n"
                        documents.append(jsobj['topk'][jdx]['text'])
                        scores.append(score)
                if len(documents) > 0:
                    query_list.append(query)
                    doc_list.append(relevant_doc)
                    document_list.append(documents)
                    score_list.append(scores)
                    if len(query_list) >= self.num_queries:
                        break
            # import ipdb; ipdb.set_trace()

        return query_list, doc_list, score_list, document_list

    # def get_task_instruction_openqa(self, question):
    #     user_prompt = (
    #             'Please answer the following question. You should think step by step to solve it.\n\n'
    #             'Provide your final answer in the format \\boxed{YOUR_ANSWER}.\n\n'
    #             f'Question:\n{question}\n\n'
    #     )
    #     return user_prompt

    def extract_search_queries(self, text):
        queries = []
        start_tag = "<|begin_search_query|>"
        end_tag = "<|end_search_query|>"

        # Keep searching until no more start tags are found
        current_pos = 0
        while current_pos < len(text):
            # Find the next start tag
            start_pos = text.find(start_tag, current_pos)
            if start_pos == -1:
                break  # No more start tags found

            # Find the end tag after the start tag
            end_pos = text.find(end_tag, start_pos)
            if end_pos == -1:
                break  # No corresponding end tag found

            # Extract the query (text between the tags)
            query_start = start_pos + len(start_tag)
            query = text[query_start:end_pos].strip()
            queries.append(query)

            # Move to position after the current end tag
            current_pos = end_pos + len(end_tag)

        return queries

    def insert_search_results(self, text, search_results):
        # Constants for tags
        query_start_tag = "<|begin_search_query|>"
        query_end_tag = "<|end_search_query|>"
        result_start_tag = "<|begin_search_result|>"
        result_end_tag = "<|end_search_result|>"

        # Track position and result index
        current_pos = 0
        result_index = 0
        new_text = ""

        while current_pos < len(text) and result_index < len(search_results):
            # Find the next query tag
            start_pos = text.find(query_start_tag, current_pos)
            if start_pos == -1:
                break  # No more query tags

            # Find the end of the query tag
            end_pos = text.find(query_end_tag, start_pos)
            if end_pos == -1:
                break  # No corresponding end tag

            # Add everything from current position up to and including the end tag
            tag_end_pos = end_pos + len(query_end_tag)
            new_text += text[current_pos:tag_end_pos]

            # Add the search result after the query
            result = search_results[result_index]
            new_text += f"\n{result_start_tag} {result} {result_end_tag}"

            # Update position and result index
            current_pos = tag_end_pos
            result_index += 1

        # Add any remaining text
        if current_pos < len(text):
            new_text += text[current_pos:]

        return new_text

    # def extract_search_queries(self, text):
    #     text = text[text.find("Gather Knowledge:")+18:]
    #     # Split the text into lines
    #     lines = text.split('\n')
    #     return lines

    # def insert_search_results(self, original_text, queries, search_results):
    #     # Create a new string to build the result
    #     result_text = original_text
    #
    #     # Make sure we have the same number of queries and results
    #     if len(queries) != len(search_results):
    #         raise ValueError("Number of queries and search results must match")
    #
    #     # For each query and its result
    #     for query, result in zip(queries, search_results):
    #         # Find the query in the original text
    #         query_pos = result_text.find(query)
    #
    #         if query_pos != -1:
    #             # Calculate the position right after the query
    #             insert_pos = query_pos + len(query)
    #
    #             # Insert the search result
    #             result_text = (
    #                     result_text[:insert_pos] +
    #                     f": {result}" +
    #                     result_text[insert_pos:]
    #             )
    #
    #     return result_text

    def generate_reasoning_structure(self, user_question: str):
        io_input = self.regen_reasoning_structure_config["prompt_template"].format(
            examples=self.regen_reasoning_structure_prompt, task=user_question)
        io_output_list = self.io.generate(
            io_input,
            num_return=self.num_child_create,
            max_tokens=256,
            stop_tokens=self.regen_reasoning_structure_config["stop_tokens"],
        )
        cleaned_io_output_list = [io_output.strip() for io_output in io_output_list]
        return cleaned_io_output_list

    def generate_search_queries(self, reasoning_structure):
        io_input = self.regen_query_generation_config["prompt_template"].format(
            examples=self.regen_query_generation_prompt, task=reasoning_structure)
        io_output_list = self.io.generate(
            io_input,
            num_return=self.num_child_create,
            max_tokens=512,
            stop_tokens=self.regen_query_generation_config["stop_tokens"],
        )
        cleaned_io_output_list = [io_output.strip() for io_output in io_output_list]
        # import ipdb; ipdb.set_trace()
        return cleaned_io_output_list

    def generate_search_queries_with_question(self, user_question, options):
        prompt = prompt_with_options.format(
            **{"question": user_question, "A": options['A'], "B": options['B'], "C": options['C'],
               "D": options['D']})
        io_input = self.regen_query_generation_config["prompt_template"].format(
            examples=self.regen_query_generation_prompt, question=prompt)
        io_output_list = self.io.generate(
            io_input,
            num_return=self.num_child_create,
            max_tokens=512,
            stop_tokens=self.regen_query_generation_config["stop_tokens"],
        )
        cleaned_io_output_list = [io_output.strip() for io_output in io_output_list]
        # import ipdb; ipdb.set_trace()
        return cleaned_io_output_list

    def generate_reasoning_thought(self, user_question: str):
        if self.dataset_name in ["Musique", "HotpotQA", "Twowiki"]:
            instruction = self.regen_reasoning_thought_prompt.format(max_searches=self.max_depth_allowed)
            user_prompt = get_task_instruction_openqa(user_question)
            io_input = instruction+user_prompt
        else:
            io_input = self.regen_reasoning_thought_prompt.format(max_searches=self.max_depth_allowed, question=user_question)
        io_output_list = self.io.generate(
            io_input,
            num_return=self.num_child_create,
            max_tokens=self.max_tokens,
            stop_tokens=[],
        )
        cleaned_io_output_list = [io_output.strip() for io_output in io_output_list]
        reasonings = []
        thoughts = []
        # import ipdb; ipdb.set_trace()
        for io_output in cleaned_io_output_list:
            if io_output.find(END_SEARCH_QUERY) != -1:
                io_output = io_output[:io_output.find(END_SEARCH_QUERY) + len(END_SEARCH_QUERY)]
            thoughts.append(io_output)
            reasonings.append(io_input+io_output)
        return reasonings, [io_input] * len(reasonings), thoughts

    def extract_between(self, text: str, start_tag: str, end_tag: str):
        pattern = re.escape(start_tag) + r"(.*?)" + re.escape(end_tag)
        matches = re.findall(pattern, text, flags=re.DOTALL)
        if matches:
            return matches[-1].strip()
        return None

    def continue_reasoning_thought(self, reasoning, prompt, thought, options):
        search_query = self.extract_between(thought, BEGIN_SEARCH_QUERY, END_SEARCH_QUERY)
        # import ipdb; ipdb.set_trace()
        if "\\box" in thought or search_query is None or "answer is" in thought.lower():
            io_output_list = self.io.generate(
                prompt,
                num_return=self.mcts_num_last_votes,
                max_tokens=self.max_tokens,
                stop_tokens=[],
            )
            cleaned_io_output_list = [io_output.strip() for io_output in io_output_list]
            if options is None:
                most_confident_answers, most_confident_answer_full_completions, confidences = self._get_most_likely_answer(cleaned_io_output_list)
                return [False], most_confident_answer_full_completions, most_confident_answers, confidences
            else:
                most_likely_choices, most_likely_answers, likelihoods = self._get_most_likely_answer_with_options(
                    cleaned_io_output_list, options)
                return [False], [most_likely_answers[0]], [most_likely_choices[0]], [likelihoods[0]]
            # import ipdb; ipdb.set_trace()
        search_result_str = self.retrieve_information(search_query, self.num_retrieval)
        if self.enable_reading_documents:
            reasoning_in_documents_prompt = get_documents_to_reasonchain_instruction(search_query, search_result_str)
            analyses = self.io.generate_deterministic(reasoning_in_documents_prompt, max_tokens=self.max_tokens, stop_tokens=[], num_return=1)[0]
            append_text = f"\n{BEGIN_SEARCH_RESULT}\n{analyses}\n{END_SEARCH_RESULT}\n"
        else:
            append_text = f"\n{BEGIN_SEARCH_RESULT}\n{search_result_str}\n{END_SEARCH_RESULT}\n"
        next_prompt = reasoning + append_text
        io_output_list = self.io.generate(
            next_prompt,
            num_return=self.num_child_create,
            max_tokens=self.max_tokens,
            stop_tokens=[],
        )
        cleaned_io_output_list = [io_output.strip() for io_output in io_output_list]
        reasonings = []
        thoughts = []
        for io_output in cleaned_io_output_list:
            if io_output.find(END_SEARCH_QUERY) != -1:
                io_output = io_output[:io_output.find(END_SEARCH_QUERY) + len(END_SEARCH_QUERY)]
            reasonings.append(next_prompt+io_output)
            thoughts.append(io_output)
        keep_reasoning = [True] * len(reasonings)
        return keep_reasoning, reasonings, [next_prompt]* len(reasonings), thoughts

    def retrieve_information(self, search_query, top_k=3):
        document = ""
        if self.retrieval_corpus == "textbook":
            request_query = "http://172.16.34.21:6000/api/search?query=" + search_query + "&k=" + str(top_k)
        elif self.retrieval_corpus == "textstat":
            request_query = "http://172.16.34.21:8000/api/search?query=" + search_query + "&k=" + str(top_k)
        elif self.retrieval_corpus == "wikipedia":
            request_query = "http://172.16.34.21:9999/api/search?query=" + search_query + "&k=" + str(top_k)
        elif self.retrieval_corpus == "medcorp":
            request_query = "http://172.16.34.21:7000/api/search?query=" + search_query + "&k=" + str(top_k)
        x = requests.get(request_query)
        jsobj = json.loads(x.text)
        for idx in range(len(jsobj['topk'])):
            document = document + jsobj['topk'][idx]['text'] + "\n"
        return document

    def implement_reasoning_structure(self, user_question: str, reasoning: str):
        search_queries = self.extract_search_queries(reasoning)
        search_results = []
        retrieved_documents = ""
        for query in search_queries:
            document = self.retrieve_information(query, self.num_retrieval)
            retrieved_documents = retrieved_documents + document

        if self.enable_reading_documents:
            io_input = self.regen_module_adaptation_config["prompt_template"].format(
                examples=self.regen_module_adaptation_prompt, documents=retrieved_documents, question=user_question)
            io_output_list = self.io.generate(
                io_input,
                num_return=1,
                max_tokens=4096,#self.max_tokens,
                stop_tokens=self.regen_module_adaptation_config["stop_tokens"],
            )
            cleaned_io_output_list = [io_output.strip() for io_output in io_output_list]
            retrieved_knowledge = cleaned_io_output_list[0]
            # import ipdb; ipdb.set_trace()
            if "No helpful information found" in retrieved_knowledge:
                retrieved_knowledge = "No helpful information found"
        else:
            retrieved_knowledge = retrieved_documents
        search_results.append(retrieved_knowledge)

        # formated_reasoning = self.insert_search_results(reasoning, search_results)
        formated_reasoning = retrieved_knowledge #+ reasoning
        # import ipdb; ipdb.set_trace()
        return [formated_reasoning], search_queries, search_results

    # def revision_reasoning_solution(self, user_question, most_likely_choices, most_likely_answers, likelihoods, options):
    #     io_input = self.regen_solution_revision_config["prompt_template"].format(examples=self.regen_solution_revision_prompt, question=user_question, sol1=most_likely_answers[0], sol2=most_likely_answers[1])
    #     io_output_list = self.io.generate_deterministic(
    #         io_input,
    #         num_return=1,
    #         max_tokens=32,
    #         stop_tokens=self.regen_solution_revision_config["stop_tokens"],
    #     )
    #     cleaned_io_output_list = [io_output.strip() for io_output in io_output_list]
    #     result = cleaned_io_output_list[0]
    #     if "candidate solution 2" in result.lower():
    #         return most_likely_choices[1], most_likely_answers[1], 1.0
    #     elif "candidate solution 1" in result.lower():
    #         return most_likely_choices[0], most_likely_answers[0], 1.0
    #     return most_likely_choices[0], most_likely_answers[0], likelihoods[0]

    def revision_reasoning_solution(self, user_question, most_likely_choices, most_likely_answers, likelihoods, options):

        # Initialize win counter for each solution
        win_counts = [0] * len(most_likely_answers)

        # Total number of possible comparisons for each solution
        total_comparisons = len(most_likely_answers) - 1

        # Compare each solution pair by pair
        for i in range(len(most_likely_answers)):
            for j in range(i + 1, len(most_likely_answers)):
                # Format the prompt for this pair comparison
                io_input = self.regen_solution_revision_config["prompt_template"].format(
                    examples=self.regen_solution_revision_prompt,
                    question=user_question,
                    sol1=most_likely_answers[i],
                    sol2=most_likely_answers[j]
                )
                # Generate the comparison result
                io_output_list = self.io.generate_deterministic(
                    io_input,
                    num_return=1,
                    max_tokens=32,
                    stop_tokens=self.regen_solution_revision_config["stop_tokens"],
                )
                cleaned_io_output = io_output_list[0].strip()
                # Determine the winner of this comparison
                if "candidate solution 2" in cleaned_io_output.lower():
                    win_counts[j] += 1
                elif "candidate solution 1" in cleaned_io_output.lower():
                    win_counts[i] += 1

        # Calculate win ratios for each solution
        win_ratios = [wins / total_comparisons for wins in win_counts]

        # Find the solution with the highest win ratio
        max_ratio = max(win_ratios)

        # In case of a tie, use the solution with highest likelihood
        if win_ratios.count(max_ratio) > 1:
            # Get indices of solutions with the max win ratio
            tied_indices = [i for i, ratio in enumerate(win_ratios) if ratio == max_ratio]
            # Find the one with highest likelihood among tied solutions
            best_index = tied_indices[0]
            best_likelihood = likelihoods[tied_indices[0]]
            for idx in tied_indices[1:]:
                if likelihoods[idx] > best_likelihood:
                    best_index = idx
                    best_likelihood = likelihoods[idx]
        else:
            # Return the solution with the highest win ratio
            best_index = win_ratios.index(max_ratio)
        return most_likely_choices[best_index], most_likely_answers[best_index], win_ratios[best_index]

    def debate_reasoning_solution(self, user_question, most_likely_choices, most_likely_answers, likelihoods, options):
        io_input = self.regen_negative_prompt.format(question=user_question, aff=most_likely_answers[0])
        io_output_list = self.io.generate(
            io_input,
            num_return=self.mcts_num_last_votes,
            max_tokens=self.max_tokens,
            stop_tokens=[],
        )
        cleaned_io_output_list = [io_output.strip() for io_output in io_output_list]
        negative_opinion = cleaned_io_output_list[0]
        io_input = self.regen_judge_prompt.format(question=user_question, aff=most_likely_answers[0],
                                                        neg=negative_opinion)
        io_output_list = self.io.generate(
            io_input,
            num_return=self.mcts_num_last_votes,
            max_tokens=self.max_tokens,
            stop_tokens=[],
        )
        cleaned_io_output_list = [io_output.strip() for io_output in io_output_list]
        if options is None:
            debate_likely_choices, debate_likely_answers, debate_likelihoods = self._get_most_likely_answer(cleaned_io_output_list)
        else:
            debate_likely_choices, debate_likely_answers, debate_likelihoods = self._get_most_likely_answer_with_options(cleaned_io_output_list, options)
        # import ipdb; ipdb.set_trace()
        return debate_likely_choices, debate_likely_answers, debate_likelihoods

    def generate_initial_hypothesis(self, user_question: str, options: dict):
        prompt = prompt_with_options.format(
            **{"question": user_question, "A": options['A'], "B": options['B'], "C": options['C'], "D": options['D']})
        io_input = self.regen_initial_hypothesis_config["prompt_template"].format(
            examples=self.regen_initial_hypothesis_prompt, question=prompt)
        io_output_list = self.io.generate(
            io_input,
            num_return=self.num_child_create,
            max_tokens=self.max_tokens,
            stop_tokens=self.regen_initial_hypothesis_config["stop_tokens"],
        )
        cleaned_io_output_list = [io_output.strip() for io_output in io_output_list]
        # import ipdb;ipdb.set_trace()
        return cleaned_io_output_list

    def generate_integration_hypothesis(self, user_question: str, initial_hypothesis, search_queries, search_results):
        if len(search_queries) == 0:
            return [initial_hypothesis]
        # retrieved_knowledge = ""
        # for query, result in zip(search_queries, search_results):
        #     retrieved_knowledge = retrieved_knowledge + query+": "+ result +"\n"
        retrieved_knowledge = search_results[0]
        io_input = self.regen_integrate_hypothesis_config["prompt_template"].format(
            examples=self.regen_integrate_hypothesis_prompt, question=user_question, hypothesis=initial_hypothesis, knowledge=retrieved_knowledge)
        io_output_list = self.io.generate(
            io_input,
            num_return=self.num_child_create,
            max_tokens=self.max_tokens,
            stop_tokens=self.regen_integrate_hypothesis_config["stop_tokens"],
        )
        cleaned_io_output_list = [io_output.strip() for io_output in io_output_list]
        # import ipdb; ipdb.set_trace()
        return cleaned_io_output_list

    def generate_reasoning_solution(self, user_question: str, reasoning_structure: str, options: dict, search_queries: List, search_results: List):
        # da_likely_answers, da_likelihoods, da_likely_choices = self.generate_direct_answers(user_question, options)
        # all_choices = [da_likely_choices[0]]
        # all_answers = [da_likely_answers[0]]
        # all_scores = [da_likelihoods[0]]
        # all_modules = [1]
        all_choices = []
        all_answers = []
        all_modules = []
        all_scores = []
        # retrieved_knowledge = ""
        # for query, result in zip(search_queries, search_results):
        #     retrieved_knowledge = retrieved_knowledge + query + ": " + result + "\n"
        retrieved_knowledge = search_results[0]
        # prompt = prompt_with_options.format(
        #     **{"question": user_question, "A": options['A'], "B": options['B'], "C": options['C'], "D": options['D']})
        content = [key + ": " + value for key, value in options.items()]
        suffix = ", ".join(content)
        prompt = user_question + " " + suffix
        # io_input = self.regen_reasoning_solution_config["prompt_template"].format(examples=self.regen_reasoning_solution_prompt, structure=reasoning_structure, question=prompt, knowledge=retrieved_knowledge)
        io_input = self.fewshot_cot_rag_config["prompt_template"].format(examples=self.fewshot_cot_rag_prompt, question=prompt, documents=retrieved_knowledge)
        io_output_list = self.io.generate(
            io_input,
            num_return=self.mcts_num_last_votes,
            max_tokens=self.max_tokens,
            stop_tokens=self.regen_reasoning_solution_config["stop_tokens"],
        )

        cleaned_io_output_list = [io_output.strip() for io_output in io_output_list]
        if options is None:
            most_likely_choices, most_likely_answers, likelihoods = self._get_most_likely_answer(cleaned_io_output_list)
        else:
            most_likely_choices, most_likely_answers, likelihoods = self._get_most_likely_answer_with_options(cleaned_io_output_list, options)
        # import ipdb; ipdb.set_trace()
        all_choices.append(most_likely_choices[0])
        all_answers.append(most_likely_answers[0])
        all_scores.append(likelihoods[0])
        all_modules.append(2)
        if self.enable_debate_solution:
            debate_likely_choices, debate_likely_answers, debate_likelihoods = self.debate_reasoning_solution(user_question, most_likely_choices, most_likely_answers, likelihoods, options)
            all_choices.append(debate_likely_choices[0])
            all_answers.append(debate_likely_answers[0])
            all_scores.append(debate_likelihoods[0])
            all_modules.append(3)
        return all_choices, all_answers, all_scores, all_modules

    def generate_reasoning_solution_with_hypothesis(self, user_question: str, hypothesis: str, options: dict, expected_answer: str):
        # da_likely_answers, da_likelihoods, da_likely_choices = self.generate_direct_answers(user_question, options)
        # all_choices = [da_likely_choices[0]]
        # all_answers = [da_likely_answers[0]]
        # all_scores = [da_likelihoods[0]]
        # all_modules = [1]
        all_choices = []
        all_answers = []
        all_modules = []
        all_scores = []
        # prompt = prompt_with_options.format(
        #     **{"question": user_question, "A": options['A'], "B": options['B'], "C": options['C'], "D": options['D']})
        content = [key + ": " + value for key, value in options.items()]
        suffix = ", ".join(content)
        prompt = user_question + " " + suffix
        io_input = self.regen_decision_solution_config["prompt_template"].format(
            examples=self.regen_decision_solution_prompt, hypothesis=hypothesis, question=prompt)
        io_output_list = self.io.generate(
            io_input,
            num_return=self.mcts_num_last_votes,
            max_tokens=self.max_tokens,
            stop_tokens=self.regen_decision_solution_config["stop_tokens"],
        )
        cleaned_io_output_list = [io_output.strip() for io_output in io_output_list]
        if options is None:
            most_likely_choices, most_likely_answers, likelihoods = self._get_most_likely_answer(cleaned_io_output_list)
        else:
            most_likely_choices, most_likely_answers, likelihoods = self._get_most_likely_answer_with_options(cleaned_io_output_list, options)
        # if most_likely_choices[0] != expected_answer:
        #     import ipdb; ipdb.set_trace()
        all_choices.append(most_likely_choices[0])
        all_answers.append(most_likely_answers[0])
        all_scores.append(likelihoods[0])
        all_modules.append(2)
        if self.enable_debate_solution:
            debate_likely_choices, debate_likely_answers, debate_likelihoods = self.debate_reasoning_solution(user_question, most_likely_choices, most_likely_answers, likelihoods, options)
            all_choices.append(debate_likely_choices[0])
            all_answers.append(debate_likely_answers[0])
            all_scores.append(debate_likelihoods[0])
            all_modules.append(3)
        return all_choices, all_answers, all_scores, all_modules

    def generate_direct_answers_with_agent_rag(self, user_question: str,  options: dict):
        decompose_query_prompt = self.decompose_query_prompt
        existing_queries_and_docs, next_query_id = "", 1
        io_input = (
                    decompose_query_prompt
                    + "\n\n"
                    + f"Question {self.question_index}: {user_question}"
                    + "\n"
                    + existing_queries_and_docs
                    + f"Query {self.question_index}.{next_query_id}:"
        )
        io_output_list = self.io.generate(
                io_input,
                max_tokens=128,
                num_return=self.num_queries,
                stop_tokens=[
                    "\n",
                    "\n\n",
                    "Document",
                    "Document ",
                    f"Document {self.question_index}.{next_query_id}",
                    f"Document {self.question_index}.{next_query_id}:",
                    f"Document {self.question_index}.{next_query_id}: ",
                ],
        )
        try:
            gen_queries = [o.strip() for o in io_output_list]
            documents = ""
            for query in gen_queries:
                document = self.retrieve_information(query, top_k=1)
                documents = documents + document + "\n"
            direct_answer_list, value_list, choice_list = [], [], []
            num_return = self.mcts_num_last_votes
            # prompt = prompt_with_options.format(
            #     **{"question": user_question, "A": options['A'], "B": options['B'], "C": options['C'],
            #        "D": options['D']})
            content = [key + ": " + value for key, value in options.items()]
            suffix = ", ".join(content)
            prompt = user_question + " " + suffix
            io_input, cleaned_io_output_list = self._fewshot_cot_rag_answer_question(
                question=prompt, documents=documents, num_return=num_return
            )
            # documents = ""
            if self.enable_self_reward:
                most_likely_choices, most_likely_answers, likelihoods = self._get_most_likely_answer_with_self_reward(
                    user_question, cleaned_io_output_list, options)
            else:
                most_likely_choices, most_likely_answers, likelihoods = self._get_most_likely_answer_with_options(cleaned_io_output_list, options)
            # most_likely_choices, most_likely_answers, likelihoods = self._get_most_likely_answer_with_options(cleaned_io_output_list, options)
        except Exception as e:
            # raise GeneratorError(
            #     source="generate direct answer from: few shot cot",
            #     io_input=io_input,
            #     io_output_list=cleaned_io_output_list,
            # )
            print("exception: ",e)
            most_likely_choices, most_likely_answers, likelihoods = None, None, None
        # import ipdb; ipdb.set_trace()
        if most_likely_choices is not None and likelihoods is not None and most_likely_answers is not None:
            direct_answer_list.extend(most_likely_answers)
            value_list.extend(likelihoods)
            choice_list.extend(most_likely_choices)
            return [direct_answer_list[0]], [value_list[0]], [choice_list[0]]
        return direct_answer_list, value_list, choice_list

    def generate_direct_answers_with_naive_rag(self, user_question: str, options: dict):
        direct_answer_list, value_list, choice_list = [], [], []

        try:
            # retrieval
            documents = self.retrieve_information(user_question, self.num_retrieval)
            # documents = self.retrieve_information(user_question, 3)
            # documents = ""
            # ! few shot cot
            num_return = self.mcts_num_last_votes
            # prompt = prompt_with_options.format(
            #     **{"question": user_question, "A": options['A'], "B": options['B'], "C": options['C'],
            #        "D": options['D']})
            content = [key + ": " + value for key, value in options.items()]
            suffix = ", ".join(content)
            prompt = user_question + " " + suffix
            io_input, cleaned_io_output_list = self._fewshot_cot_rag_answer_question(
                question=prompt, documents=documents, num_return=num_return
            )
            if self.enable_self_reward:
                most_likely_choices, most_likely_answers, likelihoods = self._get_most_likely_answer_with_self_reward(
                    user_question, cleaned_io_output_list, options)
            else:
                most_likely_choices, most_likely_answers, likelihoods = self._get_most_likely_answer_with_options(cleaned_io_output_list, options)
            # most_likely_choices, most_likely_answers, likelihoods = self._get_most_likely_answer_with_options(cleaned_io_output_list, options)
        except Exception as e:
            # raise GeneratorError(
            #     source="generate direct answer from: few shot cot",
            #     io_input=io_input,
            #     io_output_list=cleaned_io_output_list,
            # )
            print("exception: ", e)
            most_likely_choices, most_likely_answers, likelihoods = None, None, None
        if most_likely_choices is not None and likelihoods is not None and most_likely_answers is not None:
            direct_answer_list.extend(most_likely_answers)
            value_list.extend(likelihoods)
            choice_list.extend(most_likely_choices)
            return [direct_answer_list[0]], [value_list[0]], [choice_list[0]]
        return direct_answer_list, value_list, choice_list

    def generate_direct_answers(self, user_question: str, options: dict):
        direct_answer_list, value_list, choice_list = [], [], []

        try:
            # ! few shot cot
            content = [key+": "+value for key, value in options.items()]
            suffix = ", ".join(content)
            prompt = user_question +" "+ suffix
            # prompt = prompt_with_options.format(**{"question": user_question, "A": options['A'], "B": options['B'], "C": options['C'], "D": options['D']})
            io_input, cleaned_io_output_list = self._fewshot_cot_answer_question(question=prompt)
            if options is not None:
                most_likely_choices, most_likely_answers, likelihoods = self._get_most_likely_answer_with_options(cleaned_io_output_list, options)
            else:
                most_likely_choices, most_likely_answers, likelihoods = self._get_most_likely_answer(cleaned_io_output_list)
            # import ipdb; ipdb.set_trace()
        except Exception as e:
            # raise GeneratorError(
            #     source="generate direct answer from: few shot cot",
            #     io_input=io_input,
            #     io_output_list=cleaned_io_output_list,
            # )
            print("exception: ", e)
            most_likely_choices, most_likely_answers, likelihoods = None, None, None
        if most_likely_choices is not None and likelihoods is not None and most_likely_answers is not None:
            direct_answer_list.extend(most_likely_answers)
            value_list.extend(likelihoods)
            choice_list.extend(most_likely_choices)
            return [direct_answer_list[0]], [value_list[0]], [choice_list[0]]
        return direct_answer_list, value_list, choice_list


class Reasoning_MCTS_Node(MCTS_Node):
    def __init__(
        self,
        parent: "Reasoning_MCTS_Node",
        depth: int,
        node_type: Node_Type,
        verbose: bool = False,
        # --- For instantiating root node ---
        node_value: float = None,
        generator: Generator = None,
        disable_a5: bool = None,
        user_question: str = None,
        max_depth_allowed: int = None,
        disable_a1: bool = None,
        disable_a3: bool = None,
        disable_a4: bool = None,
        disable_a6: bool = None,
        disable_a7: bool = None,
        disable_a8: bool = None,
        options: dict = {},
        search_query_answer_weight: int = 1.0,
        retrieval_corpus: str = "wikipedia",
        choice: str = None,
        regen_seed_prompts: list = None,
        reasoning: str = None,
        thought: str = None,
        enable_triples_retrieval:bool=False,
        enable_documents_retrieval: bool = False,
        reasoning_modules:str=None,
        prompt: str = None,
        module_id: int=None,
        enable_direct_answer: bool = False,
        enable_system2_thinking: bool = False,
        enable_hypothesis_generation: bool = False,
        search_queries: List = [],
        search_results: List = [],
        # -----------------------------------
        # --- For instantiating REPHRASED_USER_QUESTION node ---
        rephrased_user_question: str = None,
        # ------------------------------------------------------
        expected_answer: str = None,
        # --- For instantiating DIRECT_ANSWER node ---
        direct_answer: str = None,
        # --------------------------------------------
        # --- For instantiating SUBQUESTION node ---
        subquestion: str = None,
        subanswer: str = None,
        is_new_subquestion: bool = None,
        # ------------------------------------------
        # --- For instantiating RE_SUBANSWER node ---
        re_subanswer: str = None,
        re_documents: str = None,
        # -------------------------------------------
        # --- For instantiating OST_STEP node ---
        ost_step: str = None,
        # --- For instantiating QUERY node ---
        query: str = None,
        document: str = None,
        is_new_query: bool = None,
        # ---------------------------------------
        # --- For node selection (not in sanity checks yet) ---
        enable_potential_score: bool = None,
        potential_answers: List[str] = None,
    ) -> None:
        """params:
        subquestion: the node is proposing a new subquestion
        subanswer: the answer corresponding to the new subquestion the node proposed
        re_subanswer: the node is proposing a new subanswer to the parent's subquestion
        """
        super().__init__()

        #! sanity checks
        try:
            assert depth is not None
            assert node_type is not None
            if node_value is not None:
                assert node_value > 0, breakpoint()

            if node_type is Node_Type.USER_QUESTION:
                assert depth == 0
                assert all(
                    attr is None
                    for attr in [
                        parent,
                        node_value,
                        rephrased_user_question,
                        direct_answer,
                        subquestion,
                        subanswer,
                        is_new_subquestion,
                        re_subanswer,
                        ost_step,
                    ]
                )
                assert all(
                    attr is not None
                    for attr in [generator, disable_a5, user_question, expected_answer, max_depth_allowed, disable_a1]
                )
            elif node_type is Node_Type.REPHRASED_USER_QUESTION:
                assert depth == 1
                assert all(
                    attr is None
                    for attr in [
                        node_value,
                        generator,
                        disable_a5,
                        user_question,
                        expected_answer,
                        direct_answer,
                        subquestion,
                        subanswer,
                        is_new_subquestion,
                        re_subanswer,
                        ost_step,
                        max_depth_allowed,
                        disable_a1,
                    ]
                )
                assert all(attr is not None for attr in [parent, rephrased_user_question])
            elif node_type is Node_Type.DIRECT_ANSWER or node_type is Node_Type.DIRECT_ANSWER_RAG:
                assert depth > 0
                assert all(
                    attr is None
                    for attr in [
                        generator,
                        disable_a5,
                        user_question,
                        expected_answer,
                        subquestion,
                        subanswer,
                        is_new_subquestion,
                        re_subanswer,
                        ost_step,
                        max_depth_allowed,
                        disable_a1,
                    ]
                )
                assert all(attr is not None for attr in [parent, node_value, direct_answer])
            elif node_type is Node_Type.SUBQUESTION:
                assert depth > 0
                assert all(
                    attr is None
                    for attr in [
                        generator,
                        disable_a5,
                        user_question,
                        expected_answer,
                        direct_answer,
                        re_subanswer,
                        ost_step,
                        max_depth_allowed,
                        disable_a1,
                    ]
                )
                assert all(
                    attr is not None for attr in [parent, node_value, subquestion, subanswer, is_new_subquestion]
                )
            elif node_type is Node_Type.RE_SUBANSWER or node_type is Node_Type.RE_SUBANSWER_RAG:
                assert depth > 0
                assert all(
                    attr is None
                    for attr in [
                        generator,
                        disable_a5,
                        user_question,
                        expected_answer,
                        direct_answer,
                        subquestion,
                        subanswer,
                        is_new_subquestion,
                        ost_step,
                        max_depth_allowed,
                        disable_a1,
                    ]
                )
                assert all(attr is not None for attr in [parent, node_value, re_subanswer])
            elif node_type is Node_Type.OST_STEP:
                assert depth > 0
                assert all(
                    attr is None
                    for attr in [
                        node_value,
                        generator,
                        disable_a5,
                        user_question,
                        rephrased_user_question,
                        expected_answer,
                        direct_answer,
                        subquestion,
                        subanswer,
                        is_new_subquestion,
                        re_subanswer,
                        max_depth_allowed,
                        disable_a1,
                    ]
                )
                assert all(attr is not None for attr in [parent, ost_step])
        except AssertionError:
            print(f"Instantiating node with type {node_type} failed!")
            breakpoint()
            exit()

        #! attributes
        self.parent = parent  # if parent is None, then the node is the root
        self.children: List["Reasoning_MCTS_Node"] = []
        self.depth = depth
        self.node_type = node_type
        self.node_value = node_value
        self.direct_answer = direct_answer
        self.subquestion = subquestion
        self.subanswer = subanswer
        self.is_new_subquestion = is_new_subquestion
        self.re_subanswer = re_subanswer
        self.ost_step = ost_step
        self.reasoning = reasoning
        self.module_id = module_id
        self.search_queries = search_queries
        self.search_results = search_results
        self.prompt = prompt
        self.thought = thought
        self.enable_system2_thinking = enable_system2_thinking
        self.enable_hypothesis_generation = enable_hypothesis_generation

        if parent is None:  # root
            self.verbose = verbose
            self.user_question = user_question
            self.expected_answer = expected_answer
            self.generator = generator
            # self.query_list, self.document_list = self.generator._generate_queries_for_user_question(user_question)
            self.disable_a5 = disable_a5
            self.question_index = generator.question_index
            self.max_depth_allowed = max_depth_allowed
            self.disable_a1 = disable_a1
            self.disable_a3 = disable_a3
            self.disable_a4 = disable_a4
            self.disable_a6 = disable_a6
            self.disable_a7 = disable_a7
            self.disable_a8 = disable_a8
            self.options = options
            self.search_query_answer_weight = search_query_answer_weight
            self.enable_potential_score = enable_potential_score
            self.retrieval_corpus = retrieval_corpus
            self.seed_prompts = regen_seed_prompts
            self.enable_triples_retrieval = enable_triples_retrieval
            self.enable_documents_retrieval = enable_documents_retrieval
            self.reasoning_modules = reasoning_modules
            self.enable_direct_answer = enable_direct_answer
        else:  # inherit from parent
            self.verbose = parent.verbose
            self.user_question = parent.user_question
            self.expected_answer = parent.expected_answer
            self.generator = parent.generator
            # self.query_list, self.document_list = parent.query_list, parent.document_list
            self.disable_a5 = parent.disable_a5
            self.question_index = parent.generator.question_index
            self.max_depth_allowed = parent.max_depth_allowed
            self.disable_a1 = parent.disable_a1
            self.disable_a3 = parent.disable_a3
            self.disable_a4 = parent.disable_a4
            self.disable_a6 = parent.disable_a6
            self.disable_a7 = parent.disable_a7
            self.disable_a8 = parent.disable_a8
            self.enable_potential_score = parent.enable_potential_score
            self.options = parent.options
            self.search_query_answer_weight = parent.search_query_answer_weight
            self.retrieval_corpus = parent.retrieval_corpus
            self.choice = choice
            self.seed_prompts = parent.seed_prompts
            self.enable_triples_retrieval = parent.enable_triples_retrieval
            self.reasoning_modules = parent.reasoning_modules
            self.enable_documents_retrieval = parent.enable_documents_retrieval
            self.enable_direct_answer = parent.enable_direct_answer

        #! keep track of paraphrasing
        if node_type is Node_Type.USER_QUESTION:
            self.paraphrased = False
        elif node_type is Node_Type.REPHRASED_USER_QUESTION:
            self.paraphrased = True
            self.user_question = rephrased_user_question
        else:
            assert parent is not None
            self.paraphrased = parent.paraphrased

        #! record number of subquestions till now
        if parent is None:  # root
            self.subquestion_counter = 0
            self.query_counter = 0
        else:
            if node_type is Node_Type.SUBQUESTION and is_new_subquestion:
                self.subquestion_counter = parent.subquestion_counter + 1
            else:
                self.subquestion_counter = parent.subquestion_counter

            if node_type is Node_Type.SEARCH_QUERY and is_new_query:
                self.query_counter = parent.query_counter + 1
            else:
                self.query_counter = parent.query_counter

        #! record number of one-step thought steps till now
        if parent is None:  # root
            self.ost_step_counter = 0
        else:
            if node_type is Node_Type.OST_STEP:
                self.ost_step_counter = parent.ost_step_counter + 1
            else:
                self.ost_step_counter = parent.ost_step_counter

        #! record solution trace from root to the current node. key: subquestion id
        if parent is None:  # root
            assert self.node_type is Node_Type.USER_QUESTION
            self.solution_trace: Dict[int, Dict[str, str]] = {0: {"user_question": user_question, "ost_step": {}, "query": {}}}
        else:
            assert self.node_type is not Node_Type.USER_QUESTION
            self.solution_trace = deepcopy(parent.solution_trace)

            if node_type is Node_Type.REPHRASED_USER_QUESTION:
                self.solution_trace[0]["user_question"] = rephrased_user_question
            elif node_type is Node_Type.DIRECT_ANSWER:
                assert self.subquestion_counter in self.solution_trace.keys()
                assert self.subquestion_counter == parent.subquestion_counter
                self.solution_trace[self.subquestion_counter]["direct_answer"] = {
                    "text": direct_answer,
                    "value": node_value,
                }
            elif node_type is Node_Type.SUBQUESTION:
                assert is_new_subquestion and self.subquestion_counter == parent.subquestion_counter + 1
                self.solution_trace[self.subquestion_counter] = {
                    "subquestion": subquestion,
                    "subanswer": {"text": subanswer, "value": node_value},
                    "ost_step": {},
                    "query": {}
                }
            elif node_type is Node_Type.SEARCH_QUERY:
                assert is_new_query and self.query_counter == parent.query_counter + 1
                self.solution_trace[self.subquestion_counter]["query"][self.query_counter] = {
                    "query": query,
                    "document": document,
                }
            elif node_type is Node_Type.RE_SUBANSWER or node_type is Node_Type.RE_SUBANSWER_RAG:
                assert parent.subquestion is not None
                assert self.subquestion_counter == parent.subquestion_counter
                assert self.solution_trace[self.subquestion_counter]["subquestion"] == parent.subquestion
                self.solution_trace[self.subquestion_counter]["subanswer"] = {"text": re_subanswer, "value": node_value}
            elif node_type is Node_Type.OST_STEP:
                assert "ost_step" in self.solution_trace[self.subquestion_counter].keys()
                self.solution_trace[self.subquestion_counter]["ost_step"][self.ost_step_counter] = ost_step

        #! potential_score for intermediate nodes (only used for node selection)
        if self.enable_potential_score:
            self.potential_answers = potential_answers
            self.potential_score = 0
            if parent is None:  # root
                assert self.node_type is Node_Type.USER_QUESTION
                self.potential_answers_history = {}
            else:
                assert self.node_type is not Node_Type.USER_QUESTION
                self.potential_answers_history = deepcopy(parent.potential_answers_history)
                self.potential_answers_history[self.depth] = potential_answers


    def __str__(self) -> str:
        type2str = {
            Node_Type.USER_QUESTION: "U",
            Node_Type.REPHRASED_USER_QUESTION: "RU",
            Node_Type.DIRECT_ANSWER: "DA",
            Node_Type.DIRECT_ANSWER_RAG: "DAR",
            Node_Type.SUBQUESTION: "SQ",
            Node_Type.RE_SUBANSWER: "RS",
            Node_Type.RE_SUBANSWER_RAG: "RSA",
            Node_Type.OST_STEP: "TS",
            Node_Type.SEARCH_QUERY: "SR",
        }
        return f"{type2str[self.node_type]}-{self.id}"

    def _create_children(self):

        def do_action_generate_reasoning_structure():
            reasoning_structures = self.generator.generate_reasoning_structure(
                user_question=self.user_question
            )
            for structure in reasoning_structures:
                self.children.append(
                    Reasoning_MCTS_Node(
                        parent=self,
                        depth=self.depth + 1,
                        node_type=Node_Type.REASONING_STRUCTURE,
                        reasoning=structure,
                        module_id=2,
                        enable_hypothesis_generation=self.enable_hypothesis_generation
                    )
                )

        def do_action_generate_search_queries():
            # reasoning_structures = self.generator.generate_search_queries(self.reasoning)
            reasoning_structures = self.generator.generate_search_queries_with_question(self.user_question, self.options)
            for structure in reasoning_structures:
                self.children.append(
                    Reasoning_MCTS_Node(
                        parent=self,
                        depth=self.depth + 1,
                        node_type=Node_Type.QUERY_GENERATION,
                        reasoning=structure,
                        module_id=2,
                        enable_hypothesis_generation=self.enable_hypothesis_generation
                    )
                )

        def do_action_generate_reasoning_thought():
            reasoning_thoughts, prompts, thoughts = self.generator.generate_reasoning_thought(
                user_question=self.user_question
            )
            for structure, prompt, thought in zip(reasoning_thoughts, prompts, thoughts):
                self.children.append(
                    Reasoning_MCTS_Node(
                        parent=self,
                        depth=self.depth + 1,
                        node_type=Node_Type.REASONING_THOUGHT,
                        reasoning=structure,
                        prompt=prompt,
                        thought=thought,
                        module_id=2
                    )
                )

        def do_action_continue_reasoning_thought():
            keep_reasonings, reasonings, reasoning_answers, reasoning_scores = self.generator.continue_reasoning_thought(reasoning=self.reasoning, prompt=self.prompt, thought=self.thought, options=self.options)
            for keep, reasoning, answer, score in zip(keep_reasonings, reasonings, reasoning_answers, reasoning_scores):
                if keep:
                    self.children.append(
                        Reasoning_MCTS_Node(
                            parent=self,
                            depth=self.depth + 1,
                            node_type=Node_Type.REASONING_THOUGHT,
                            reasoning=reasoning,
                            prompt=answer,
                            thought=score,
                            module_id=2
                        )
                    )
                else:
                    self.children.append(
                        Reasoning_MCTS_Node(
                            parent=self,
                            depth=self.depth + 1,
                            node_type=Node_Type.DIRECT_ANSWER,
                            node_value=score,
                            direct_answer=answer,
                            reasoning=reasoning,
                            choice=answer,
                            module_id=2
                        )
                    )

        def do_action_implement_reasoning_structure():
            reasoning_plans, search_queries, search_results = self.generator.implement_reasoning_structure(
                user_question=self.user_question, reasoning=self.reasoning
            )
            for plan in reasoning_plans:
                self.children.append(
                    Reasoning_MCTS_Node(
                        parent=self,
                        depth=self.depth + 1,
                        node_type=Node_Type.REASONING_PLAN,
                        reasoning=plan,
                        module_id=self.module_id,
                        search_queries=search_queries,
                        search_results=search_results,
                        enable_hypothesis_generation=self.enable_hypothesis_generation
                    )
                )

        def do_action_generate_reasoning_solution():
            most_likely_choices, most_likely_answers, likelihoods, modules = self.generator.generate_reasoning_solution(
                user_question=self.user_question, reasoning_structure=self.parent.parent.reasoning, options=self.options, search_queries=self.search_queries, search_results=self.search_results
            )
            for direct_answer, value, solution, module_id in zip(most_likely_choices, likelihoods, most_likely_answers, modules):
                if self.options is not None:
                    self.children.append(
                        Reasoning_MCTS_Node(
                            parent=self,
                            depth=self.depth + 1,
                            node_type=Node_Type.DIRECT_ANSWER,
                            node_value=value,
                            direct_answer=solution,
                            choice=direct_answer,
                            reasoning=solution,
                            module_id=module_id,
                            search_queries=self.search_queries,
                            search_results=self.search_results
                        ))

        def do_action_generate_initial_hypothesis():
            hypotheses = self.generator.generate_initial_hypothesis(
                user_question=self.user_question, options=self.options
            )
            for hypo in hypotheses:
                self.children.append(
                    Reasoning_MCTS_Node(
                        parent=self,
                        depth=self.depth + 1,
                        node_type=Node_Type.INITIAL_HYPOTHESIS,
                        reasoning=hypo,
                        module_id=self.module_id,
                        search_queries=self.search_queries,
                        search_results=self.search_results,
                        enable_hypothesis_generation=self.enable_hypothesis_generation
                    )
                )

        def do_action_generate_integration_hypothesis():
            hypotheses = self.generator.generate_integration_hypothesis(
                self.user_question, self.reasoning, self.search_queries, self.search_results
            )
            for hypo in hypotheses:
                self.children.append(
                    Reasoning_MCTS_Node(
                        parent=self,
                        depth=self.depth + 1,
                        node_type=Node_Type.INTEGRATION_HYPOTHESIS,
                        reasoning=hypo,
                        module_id=self.module_id,
                        search_queries=self.search_queries,
                        search_results=self.search_results,
                    )
                )

        def do_action_generate_reasoning_solution_with_hypothesis():
            most_likely_choices, most_likely_answers, likelihoods, modules = self.generator.generate_reasoning_solution_with_hypothesis(
                user_question=self.user_question, hypothesis=self.reasoning, options=self.options, expected_answer=self.expected_answer
            )
            for direct_answer, value, solution, module_id in zip(most_likely_choices, likelihoods, most_likely_answers, modules):
                if self.options is not None:
                    self.children.append(
                        Reasoning_MCTS_Node(
                            parent=self,
                            depth=self.depth + 1,
                            node_type=Node_Type.DIRECT_ANSWER,
                            node_value=value,
                            direct_answer=solution,
                            choice=direct_answer,
                            reasoning=solution,
                            module_id=module_id,
                            search_queries=self.search_queries,
                            search_results=self.search_results
                        ))

        def do_action_generate_direct_answers():
            #! ACTION: generate direct answer for the user question (w/ or w/o hint)
            (direct_answer_list, value_list, choice_list) = self.generator.generate_direct_answers(
                user_question=self.user_question, options=self.options
            )
            # import ipdb; ipdb.set_trace()
            for direct_answer, value, choice in zip(direct_answer_list, value_list, choice_list):
                if self.options is not None:
                    self.children.append(
                        Reasoning_MCTS_Node(
                            parent=self,
                            depth=self.depth + 1,
                            node_type=Node_Type.DIRECT_ANSWER,
                            node_value=value,
                            direct_answer=direct_answer,
                            reasoning=direct_answer,
                            choice=choice,
                            module_id=1
                        )
                    )
                else:
                    self.children.append(
                        Reasoning_MCTS_Node(
                            parent=self,
                            depth=self.depth + 1,
                            node_type=Node_Type.DIRECT_ANSWER,
                            node_value=value,
                            direct_answer=direct_answer,
                            reasoning=direct_answer,
                            module_id=1
                        )
                    )

        def do_action_generate_direct_answers_with_naive_rag():
            (rag_answer_list, rag_value_list, rag_choice_list) = self.generator.generate_direct_answers_with_naive_rag(
                user_question=self.user_question, options=self.options
            )
            for direct_answer, value, choice in zip(rag_answer_list, rag_value_list, rag_choice_list):
                if np.isnan(value) or value <= 0:
                    continue
                self.children.append(
                    Reasoning_MCTS_Node(
                        parent=self,
                        depth=self.depth + 1,
                        node_type=Node_Type.DIRECT_ANSWER,
                        reasoning=direct_answer,
                        node_value=value,
                        direct_answer=direct_answer,
                        choice=choice,
                        module_id=2
                    )
                )

        def do_action_generate_direct_answers_with_agent_rag():
            (rag_answer_list, rag_value_list, rag_choice_list) = self.generator.generate_direct_answers_with_agent_rag(
                user_question=self.user_question, options=self.options
            )
            # import ipdb; ipdb.set_trace()
            for direct_answer, value, choice in zip(rag_answer_list, rag_value_list, rag_choice_list):
                if np.isnan(value) or value <= 0:
                    continue
                self.children.append(
                    Reasoning_MCTS_Node(
                        parent=self,
                        depth=self.depth + 1,
                        node_type=Node_Type.DIRECT_ANSWER,
                        reasoning=direct_answer,
                        node_value=value,
                        direct_answer=direct_answer,
                        choice=choice,
                        module_id=3
                    )
                )

        #! create children
        if self.node_type is Node_Type.USER_QUESTION:
            # do_action_generate_reasoning_thought()
            if not self.disable_a7:
                do_action_generate_direct_answers_with_naive_rag()
            if not self.disable_a6:
                do_action_generate_direct_answers_with_agent_rag()
            if self.enable_system2_thinking:
                do_action_generate_reasoning_structure()
            if self.enable_direct_answer:
                do_action_generate_direct_answers()
                # import ipdb; ipdb.set_trace()
                # self.enable_direct_answer = False
        elif self.node_type is Node_Type.REASONING_THOUGHT:
            do_action_continue_reasoning_thought()
        elif self.node_type is Node_Type.REASONING_STRUCTURE:
            do_action_generate_search_queries()
        elif self.node_type is Node_Type.QUERY_GENERATION:
            do_action_implement_reasoning_structure()
        elif self.node_type is Node_Type.REASONING_PLAN:
            # import ipdb; ipdb.set_trace()
            if self.enable_hypothesis_generation:
                do_action_generate_initial_hypothesis()
            else:
                do_action_generate_reasoning_solution()
        elif self.node_type is Node_Type.INITIAL_HYPOTHESIS:
            do_action_generate_integration_hypothesis()
        elif self.node_type is Node_Type.INTEGRATION_HYPOTHESIS:
            do_action_generate_reasoning_solution_with_hypothesis()

        # assert self.children
        return self.children

    def is_valid_leaf_node(self):
        #! a valid solution can only be in SUBQUESTION type or DIRECT_ANSWER type
        return self.node_type is Node_Type.DIRECT_ANSWER

    def is_valid_solution_node(self):
        #! a valid solution can only be in SUBQUESTION type or DIRECT_ANSWER type or OST_STEP type
        return self.node_type is Node_Type.DIRECT_ANSWER

    def set_potential_score(self, score: float):
        self.potential_score = score

    def find_children(self, rollout_id: int):
        self.children = self.children or self._create_children()
        for child in self.children:
            child.set_rollout_id(rollout_id)
        # assert self.children
        return self.children

    def is_terminal(self):
        return self.depth >= self.max_depth_allowed or self.is_valid_leaf_node()

    def calculate_reward(self):
        if self.is_valid_leaf_node():
            assert self.node_value is not None, breakpoint()
            return self.node_value
        else:
            return 0

    def skip_backprop(self):
        return self.node_type is Node_Type.USER_QUESTION or self.node_type is Node_Type.REPHRASED_USER_QUESTION

    def get_choice(self):
        if self.choice is not None:
            return self.choice
        # import ipdb; ipdb.set_trace()
        if self.node_type is Node_Type.SUBQUESTION:
            completion = self.subanswer
        elif self.node_type is Node_Type.DIRECT_ANSWER or self.node_type is Node_Type.DIRECT_ANSWER_RAG:
            completion = self.direct_answer
        else:
            return None
        if self.options is None:
            return None
        answer = self.generator.evaluator.extract_answer_from_model_completion(completion)
        valid, choice = self.generator.evaluator.check_valid_answer(answer, completion, self.options, 0)
        if valid:
            return choice
        return None


def search_for_answers(args, user_question: str, question_id: int, gt_answer: str, generator: Generator, options: dict, seed_prompts: list):
    verbose_print(
        f"********************* Searching for answers to question {question_id} ********************* ", args.verbose
    )

    #! build an MCTS searcher
    mcts_searcher = MCTS_Searcher(
        exploration_weight=args.mcts_exploration_weight,
        weight_scheduler=args.mcts_weight_scheduler,
        num_rollouts=args.num_rollouts,
        discount=args.mcts_discount_factor,
        verbose=args.verbose,
    )


    #! build the MCTS tree
    root_node = Reasoning_MCTS_Node(
        parent=None,
        depth=0,
        node_type=Node_Type.USER_QUESTION,
        verbose=args.verbose,
        generator=generator,
        disable_a5=args.disable_a5,
        user_question=user_question,
        expected_answer=gt_answer,
        max_depth_allowed=args.max_depth_allowed,
        disable_a1=args.disable_a1,
        disable_a3=args.disable_a3,
        disable_a4=args.disable_a4,
        disable_a6=args.disable_a6,
        disable_a7=args.disable_a7,
        disable_a8=args.disable_a8,
        options=options,
        search_query_answer_weight= args.search_query_weight,
        enable_potential_score=args.enable_potential_score,
        retrieval_corpus=args.retrieval_corpus,
        regen_seed_prompts=seed_prompts,
        enable_triples_retrieval=args.enable_triples_retrieval,
        reasoning_modules=args.reasoning_modules,
        enable_documents_retrieval=args.enable_documents_retrieval,
        enable_direct_answer=args.enable_direct_answer,
        enable_system2_thinking=args.enable_system2_thinking,
        enable_hypothesis_generation=args.enable_hypothesis_generation
    )

    model_rollout_nodes = []
    for i in (pbar := trange(args.num_rollouts, disable=True, position=0)):
        rollout_node = mcts_searcher.do_rollout(root_node, i)
        model_rollout_nodes.append(rollout_node)
        best_choice, freq_choice, choice_info, all_solution_nodes, all_solutions = stochastic_find_best_solution(
            root_node, generator.evaluator, enable_potential_score=args.enable_potential_score
        )
        if len(all_solution_nodes) >= args.num_rollouts:
            break
    return best_choice, freq_choice, choice_info, all_solution_nodes, all_solutions
        # if args.save_tree:
        #     with open(
        #         os.path.join(
        #             args.answer_sheets_dir,
        #             f"Question {question_id:04d} - Rollout {i}.tree",
        #         ),
        #         "w",
        #     ) as f:
        #         print_tree_from_root(
        #             mcts_searcher=mcts_searcher,
        #             rollout_id=i,
        #             root_node=root_node,
        #             chosen_node=chosen_node,
        #             file=f,
        #         )

    #! record final traces
    # js = [{"trace": node.solution_trace, "rollout_id": node.rollout_id} for node in all_solution_nodes]
    # with open(os.path.join(args.answer_sheets_dir, f"Question {question_id:04d} - Final Solutions.json"), "w") as f:
    #     json.dump(js, f)
    #
    # js2 = [{"trace": node.solution_trace, "rollout_id": i} for i, node in enumerate(model_rollout_nodes)]
    # with open(os.path.join(args.answer_sheets_dir, f"Question {question_id:04d} - Rollout Solutions.json"), "w") as f:
    #     json.dump(js2, f)
    #
    # if args.enable_potential_score:
    #     js = [node.potential_answers_history for node in all_solution_nodes]
    #     with open(os.path.join(args.answer_sheets_dir, f"Question {question_id:04d} - Potentials.json"), "w") as f:
    #         json.dump(js, f)

    # return model_solutions, i, model_all_solutions[-1], all_solution_nodes