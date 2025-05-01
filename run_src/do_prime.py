# Licensed under the MIT license.

import sys
import os, json, time
from tqdm import tqdm

sys.path.append(".")
# sys.path.append("/home/htran/generation/med_preferences/rStar")
# sys.path.append("run_src")

from common.utils import fix_seeds, setup_model_parallel, read_json
# from common.utils import *
from common.arguments import get_parser, post_process_args, save_args
from run_src.rstar_utils import GeneratorError
from run_src.rstar_utils import Node_Type
from Multi_agents_for_reasoning_prime import Generator, search_for_answers
from eval_src.Evaluator import *
from evaluate import run_evaluation
import evaluate


def get_pattern(solution_node):
    action_trace = []
    node = solution_node
    while node.node_type is not Node_Type.USER_QUESTION:
        if node.node_type is Node_Type.OST_STEP:
            action_trace.append("A1")
        elif node.node_type is Node_Type.DIRECT_ANSWER:
            action_trace.append("A2")
        elif node.node_type is Node_Type.SUBQUESTION:
            action_trace.append("A3")
        elif node.node_type is Node_Type.RE_SUBANSWER:
            action_trace.append("A4")
        elif node.node_type is Node_Type.DIRECT_ANSWER_RAG:
            action_trace.append("A6")
        elif node.node_type is Node_Type.RE_SUBANSWER_RAG:
            action_trace.append("A7")
        node = node.parent
    action_trace.reverse()
    return action_trace

def get_trace(solution_node):
    trace = []
    node = solution_node
    while node.node_type is not Node_Type.USER_QUESTION:
        trace.append(node.reasoning)
        node = node.parent
    trace.reverse()
    return trace


def main(args):
    fix_seeds(args.seed)
    if args.model_parallel:
        args.local_rank, args.world_size = setup_model_parallel()
    else:
        args.local_rank, args.world_size = 0, 1

    test_file = os.path.join(args.data_root, args.dataset_name, args.test_json_filename + ".json")
    assert os.path.exists(test_file), f"Test file {test_file} does not exist."
    data_item_list = read_json(test_file)
    start_from = args.start_from
    if args.enable_revision and args.revision_file_path is not None and os.path.exists(args.revision_file_path):
        with open(args.revision_file_path, "r") as f:
            sys1_data = json.load(f)

    if os.path.exists(args.save_path):
        with open(args.save_path, "r") as f:
            save_generations = json.load(f)
            if start_from != 0:
                save_generations = save_generations[:start_from]
            num_tested = save_generations[-1]['num_tested']
            if "options" in data_item_list[0].keys():
                num_correct = save_generations[-1]['num_correct']
                num_correct_gen = save_generations[-1]['num_gen_correct']
                correct_modules = save_generations[-1]['correct_modules']
                freq_correct_modules = save_generations[-1]['freq_correct_modules']
            else:
                modules_output = save_generations[-1]["modules_output"]
    else:
        save_generations = []
        num_tested = 0
        num_correct = 0
        num_correct_gen = 0
        correct_modules = {}
        modules_output = {}
        freq_correct_modules = {}

    evaluator = eval(f"{args.dataset_name}Evaluator()")

    tokenizer, model = None, None
    if args.api == "huggingface":
        from models.HuggingFace_API import load_HF_model

        tokenizer, model = load_HF_model(args.model_ckpt)
    elif args.api == "vllm":
        from models.vLLM_API import load_vLLM_model
        tokenizer, model = load_vLLM_model(args.model_ckpt, args.seed, args.tensor_parallel_size, args.half_precision)
    elif args.api == "gpt3.5-turbo":
        from models.OpenAI_API import load_OpenAI_model
        tokenizer, model = load_OpenAI_model(args.model_ckpt)
    elif args.api == "llama":
        from models.Llama_API import load_OpenAI_model
        tokenizer, model = load_OpenAI_model(args.model_ckpt)

    generator = Generator(args, tokenizer, model, evaluator)
    seed_prompts = read_json(args.regen_seed_prompts_path)["prompts"]

    for i, data_item in enumerate(
        (pbar := tqdm(data_item_list, disable=args.local_rank > 0 or args.verbose, position=1))
    ):
        if i < len(save_generations):
            continue
        if i < args.start_from:
            continue
        if "options" in data_item.keys() and "answer" in data_item.keys():
            problem_id, problem, gt_solution, options, gold_answer = data_item["id"], data_item["question"], data_item["solution"], data_item["options"], data_item['answer']
        else:
            problem_id, problem, gold_answer = data_item["id"], data_item["question"], data_item["answer"]
            options = None
            gt_solution = gold_answer
        num_tested += 1

        if args.enable_revision and args.revision_file_path is not None:
            sys1_choice = sys1_data[i]['best_choice']
            sys1_score = sys1_data[i]['choices_info'][sys1_choice]['scores'][0]

        if args.enable_revision and sys1_score >= args.revision_threshold:
            best_choice, freq_choice, choice_info, all_solutions = sys1_data[i]['best_choice'], sys1_data[i]['frequent_choice'], sys1_data[i]['choices_info'], sys1_data[i]['all_solutions']
            all_solution_nodes = None
        else:
            best_choice, freq_choice, choice_info, all_solution_nodes, all_solutions = search_for_answers(
                args=args, user_question=problem, question_id=i, gt_answer=gold_answer, generator=generator,
                options=options, seed_prompts=seed_prompts
            )
            if args.enable_revision:
                best_score = choice_info[best_choice]['scores'][0]
                if best_score < sys1_score:
                    best_choice = sys1_choice

        call_counter = generator.io.call_counter
        token_counter = generator.io.token_counter
        save_obj = {}
        save_obj['question'] = problem
        save_obj['id'] = problem_id
        save_obj['gold_solution'] = gt_solution
        save_obj['gold_answer'] = gold_answer
        save_obj['all_solutions'] = all_solutions
        save_obj["call_counter"] = call_counter
        save_obj["token_counter"] = token_counter
        save_obj['num_tested'] = num_tested

        if options is not None:
            correct = best_choice == gold_answer
            correct_gen = freq_choice == gold_answer
            if all_solution_nodes is not None:
                modules_choice = {}
                modules_count = {}
                for node in all_solution_nodes:
                    module_id = str(node.module_id)
                    choice = node.choice
                    if module_id not in modules_choice.keys():
                        modules_choice[module_id] ={"top_choice": choice, "score": node.node_value, "freq": 1, "freq_choice": choice}
                        modules_count[module_id] = {choice: 1}
                    else:
                        if choice not in modules_count[module_id].keys():
                            modules_count[module_id][choice] = 1
                        else:
                            modules_count[module_id][choice] += 1
                        if modules_choice[module_id]["score"] < node.node_value:
                            modules_choice[module_id]["score"] = node.node_value
                            modules_choice[module_id]["top_choice"] = node.choice
                        if modules_count[module_id][choice] > modules_choice[module_id]['freq']:
                            modules_choice[module_id]['freq'] = modules_count[module_id][choice]
                            modules_choice[module_id]['freq_choice'] = choice
                for module_id in modules_choice.keys():
                    top_choice = modules_choice[module_id]["top_choice"]
                    freq_choice = modules_choice[module_id]["freq_choice"]
                    if module_id not in correct_modules.keys():
                        correct_modules[module_id] = []
                    if module_id not in freq_correct_modules.keys():
                        freq_correct_modules[module_id] = []
                    if top_choice == gold_answer:
                        correct_modules[module_id].append(1)
                    else:
                        correct_modules[module_id].append(0)
                    if freq_choice == gold_answer:
                        freq_correct_modules[module_id].append(1)
                    else:
                        freq_correct_modules[module_id].append(0)
                if args.enable_revision and args.revision_file_path is not None:
                    if "old" not in correct_modules.keys():
                        correct_modules["old"] = []
                    if sys1_choice == gold_answer:
                        correct_modules["old"].append(1)
                    else:
                        correct_modules["old"].append(0)
                module_acc = {key: sum(correct_modules[key])/ len(correct_modules[key]) for key in correct_modules.keys()}
                freq_module_acc =  {key: sum(freq_correct_modules[key])/ len(freq_correct_modules[key]) for key in freq_correct_modules.keys()}
                save_obj["modules_choice"] = modules_choice
                save_obj["modules_count"] = modules_count
                print("top: ", module_acc)#,"freq: ", freq_module_acc)
            if correct:
                num_correct += 1
            if correct_gen:
                num_correct_gen += 1
            accuracy = num_correct / num_tested
            gen_accuracy = num_correct_gen / num_tested
            save_obj['options'] = options
            save_obj["best_choice"] = best_choice
            save_obj["frequent_choice"] = freq_choice
            save_obj["choices_info"] = choice_info
            save_obj['num_correct'] = num_correct
            save_obj['num_gen_correct'] = num_correct_gen
            save_obj['accuracy'] = accuracy
            save_obj['correct_modules'] = correct_modules
            save_obj["freq_correct_modules"] = freq_correct_modules
            print("Score accuracy: ", accuracy, "Frequent accuracy: ", gen_accuracy, "num tested: ", num_tested, "avg call: ", call_counter/num_tested, "avg token: ", token_counter/num_tested)
        else:
            highest_score = -1.0
            for idx, node in enumerate(all_solution_nodes):
                module_id = str(idx+1)
                if module_id not in modules_output.keys():
                    modules_output[module_id] = [node.reasoning]
                else:
                    modules_output[module_id].append(node.reasoning)
                if node.node_value > highest_score:
                    highest_score = node.node_value
                    top_output = node.reasoning
            if "top" not in modules_output.keys():
                modules_output["top"] = [top_output]
            else:
                modules_output["top"].append(top_output)
            modules_metrics = {}
            for key in modules_output.keys():
                filtered_data = data_item_list[:len(modules_output[key])]
                metrics = run_evaluation(filtered_data, modules_output[key])
                modules_metrics[key] = metrics
            save_obj["modules_metrics"] = modules_metrics
            save_obj["modules_output"] = modules_output
            print(modules_metrics)
            # import ipdb; ipdb.set_trace()
            # metrics = run_evaluation()
        save_generations.append(save_obj)
        with open(args.save_path, "w") as f:
            json.dump(save_generations, f)



if __name__ == "__main__":
    #! -------------------------------- Arguments --------------------------------
    parser = get_parser()

    parser.add_argument("--num_rollouts", type=int, default=15)
    parser.add_argument(
        "--num_subquestions", type=int, default=3, help="Number of trials for proposing the next subquestion"
    )
    parser.add_argument(
        "--num_queries", type=int, default=None, help="Number of trials for proposing the next query"
    )
    parser.add_argument(
        "--num_retrieval", type=int, default=None, help="Number of documents retrieval"
    )
    parser.add_argument("--num_votes", type=int, default=10)
    parser.add_argument("--max_depth_allowed", type=int, default=7)
    parser.add_argument("--num_child_create", type=int, default=1)

    # MCTS
    parser.add_argument("--num_last_votes", type=int, default=None)
    parser.add_argument("--start_from", type=int, default=0)
    parser.add_argument("--save_tree", action="store_true")
    parser.add_argument("--search_query_weight", type=float, default=0.5)
    parser.add_argument("--combine_distributions", type=str, default="add")
    parser.add_argument("--majority_threshold", type=float, default=0.5)
    parser.add_argument("--enable_majority", type=bool, default=False)
    parser.add_argument("--retrieval_threshold", type=float, default=0.0)
    parser.add_argument("--enable_chat_template", type=bool, default=False)
    parser.add_argument("--enable_triples_retrieval", type=bool, default=False)
    parser.add_argument("--enable_documents_retrieval", type=bool, default=False)
    parser.add_argument("--enable_self_reward", type=bool, default=False)
    parser.add_argument("--enable_direct_answer", type=bool, default=False)


    # Action1: Propose an one-step thought.
    parser.add_argument("--num_a1_steps", type=int, default=None)
    parser.add_argument("--disable_a1", action="store_true")
    parser.add_argument("--disable_a3", action="store_true")
    parser.add_argument("--disable_a4", action="store_true")
    parser.add_argument("--disable_a6", action="store_true")
    parser.add_argument("--disable_a7", action="store_true")
    parser.add_argument("--disable_a8", action="store_true")

    parser.add_argument("--enable_answer_revision", type=bool, default=False)
    parser.add_argument("--enable_reading_documents", type=bool, default=False)
    parser.add_argument("--enable_debate_solution", type=bool, default=False)
    parser.add_argument("--enable_system2_thinking", type=bool, default=False)
    parser.add_argument("--enable_hypothesis_generation", type=bool, default=False)
    parser.add_argument("--enable_question_decomposition", type=bool, default=False)
    parser.add_argument("--enable_revision", type=bool, default=False)
    parser.add_argument("--revision_file_path", type=str, default=None)
    parser.add_argument("--revision_threshold", type=float, default=None)


    # Paraphrasing
    parser.add_argument("--modify_prompts_for_rephrasing", action="store_true")
    parser.add_argument("--disable_a5", action="store_true")


    #! -------------------------- Used for selecting answer --------------------------
    parser.add_argument("--enable_potential_score", action="store_true")
    parser.add_argument("--disable_answer_selection", action="store_true")
    parser.add_argument("--save_generation", action="store_true")
    parser.add_argument("--save_path", type=str, default="save/results.json")

    parser.add_argument("--retrieval_corpus", type=str, default=None)
    parser.add_argument("--reasoning_modules", type=str, default="0")


    #! -------------------------------------------------------------------------------

    args = parser.parse_args()

    if args.num_last_votes is None:
        if args.enable_self_reward:
            args.num_last_votes = 10
        else:
            args.num_last_votes = 32

    if not args.disable_a1:
        if args.num_a1_steps is None:
            args.num_a1_steps = 3

    #! ----------------------------------------------------------------------------

    prompts_dir = os.path.join(args.prompts_root, args.dataset_name)

    args.fewshot_cot_prompt_path = os.path.join(prompts_dir, "fewshot_cot", "fewshot_cot_prompt.txt")
    args.fewshot_cot_config_path = os.path.join(prompts_dir, "fewshot_cot", "fewshot_cot_config.json")

    args.fewshot_cot_rag_prompt_path = os.path.join(prompts_dir, "fewshot_cot", "fewshot_cot_rag_prompt.txt")
    args.fewshot_cot_rag_config_path = os.path.join(prompts_dir, "fewshot_cot", "fewshot_cot_rag_config.json")

    args.fewshot_self_reward_prompt_path = os.path.join(prompts_dir, "fewshot_cot", "fewshot_self_reward_prompt.txt")
    args.fewshot_self_reward_config_path = os.path.join(prompts_dir, "fewshot_cot", "fewshot_self_reward_config.json")

    args.fewshot_ost_prompt_path = os.path.join(prompts_dir, "fewshot_ost", "fewshot_ost_prompt.txt")
    args.fewshot_ost_config_path = os.path.join(prompts_dir, "fewshot_ost", "fewshot_ost_config.json")

    args.decompose_template_path = os.path.join(prompts_dir, "decompose", "decompose_template.json")
    args.decompose_prompt_path = os.path.join(prompts_dir, "decompose", "decompose_prompt.txt")

    args.decompose_query_template_path = os.path.join(prompts_dir, "decompose", "decompose_query_template.json")
    args.decompose_query_prompt_path = os.path.join(prompts_dir, "decompose", "decompose_query_prompt.txt")

    args.regen_seed_prompts_path = os.path.join(prompts_dir, "regen", "seed_prompts.json")
    args.regen_module_adaptation_config_path = os.path.join(prompts_dir, "regen", "reading_documents_config.json")
    args.regen_module_adaptation_prompt_path = os.path.join(prompts_dir, "regen", "reading_documents_prompt.txt")

    args.regen_reasoning_structure_config_path = os.path.join(prompts_dir, "regen", "decompose_plan_config.json")
    args.regen_reasoning_structure_prompt_path = os.path.join(prompts_dir, "regen", "decompose_plan_prompt.txt")

    args.regen_query_generation_config_path = os.path.join(prompts_dir, "regen", "query_generation_config.json")
    args.regen_query_generation_prompt_path = os.path.join(prompts_dir, "regen", "query_generation_prompt.txt")

    args.regen_reasoning_solution_config_path = os.path.join(prompts_dir, "regen", "plan_implementation_config.json")
    args.regen_reasoning_solution_prompt_path = os.path.join(prompts_dir, "regen", "plan_implementation_prompt.txt")

    args.regen_solution_revision_config_path = os.path.join(prompts_dir, "regen", "solution_revision_config.json")
    args.regen_solution_revision_prompt_path = os.path.join(prompts_dir, "regen", "solution_revision_prompt.txt")

    args.regen_reasoning_thought_prompt_path = os.path.join(prompts_dir, "regen", "reasoning_thought_prompt.txt")

    args.regen_affirmative_prompt_path = os.path.join(prompts_dir, "regen", "debate_affirmative_prompt.txt")
    args.regen_negative_prompt_path = os.path.join(prompts_dir, "regen", "debate_negative_prompt.txt")
    args.regen_judge_prompt_path = os.path.join(prompts_dir, "regen", "debate_judge_prompt.txt")

    args.regen_initial_hypothesis_config_path = os.path.join(prompts_dir, "regen", "initial_hypothesis_config.json")
    args.regen_initial_hypothesis_prompt_path = os.path.join(prompts_dir, "regen", "initial_hypothesis_prompt.txt")

    args.regen_integrate_hypothesis_config_path = os.path.join(prompts_dir, "regen", "integration_hypothesis_config.json")
    args.regen_integrate_hypothesis_prompt_path = os.path.join(prompts_dir, "regen", "integration_hypothesis_prompt.txt")

    args.regen_decision_solution_config_path = os.path.join(prompts_dir, "regen", "decision_solution_config.json")
    args.regen_decision_solution_prompt_path = os.path.join(prompts_dir, "regen", "decision_solution_prompt.txt")

    if not args.disable_a5:
        args.rephrasing_prompt_template_path = os.path.join(prompts_dir, "rephrasing_prompt_template.txt")
        if args.modify_prompts_for_rephrasing:
            args.fewshot_cot_prompt_rephrased_path = os.path.join(
                prompts_dir, "fewshot_cot", "fewshot_cot_prompt_rephrased.txt"
            )
            args.fewshot_ost_prompt_rephrased_path = os.path.join(
                prompts_dir, "fewshot_ost", "fewshot_ost_prompt_rephrased.txt"
            )
            args.decompose_prompt_rephrased_path = os.path.join(
                prompts_dir, "decompose", "decompose_prompt_rephrased.txt"
            )
        else:
            args.fewshot_cot_prompt_rephrased_path = os.path.join(prompts_dir, "fewshot_cot", "fewshot_cot_prompt.txt")
            args.fewshot_ost_prompt_rephrased_path = os.path.join(prompts_dir, "fewshot_ost", "fewshot_ost_prompt.txt")
            args.decompose_prompt_rephrased_path = os.path.join(prompts_dir, "decompose", "decompose_prompt.txt")

    args = post_process_args(args)
    print(args)
    save_args(args)
    suffix = "ro" + str(args.num_rollouts)

    if args.enable_question_decomposition:
        suffix = suffix + "_decompose"
        args.fewshot_cot_prompt_path = os.path.join(prompts_dir, "fewshot_cot", "fewshot_cot_decompose_prompt.txt")
        args.fewshot_cot_config_path = os.path.join(prompts_dir, "fewshot_cot", "fewshot_cot_decompose_config.json")


    # if not args.disable_a6 or not args.disable_a7:

    # if args.start_from != 0:
    #     suffix = suffix + "_st" + str(args.start_from)
    if args.num_retrieval is not None:
        suffix = suffix + "_r" + str(args.num_retrieval)
    if args.retrieval_corpus is not None:
        suffix = suffix + "_" + args.retrieval_corpus

    if args.enable_chat_template:
        suffix = suffix + "_chat"
    if args.enable_direct_answer:
        suffix = suffix + "_da"
    if args.enable_reading_documents:
        suffix = suffix + "_reading"
    if args.enable_debate_solution:
        suffix = suffix + "_debate"
    if args.enable_system2_thinking:
        suffix = suffix + "_sys2"
    if args.enable_hypothesis_generation:
        suffix = suffix + "_hypo"
    if args.enable_revision:
        suffix = suffix + "_revision"
        suffix = suffix + "_t" + str(int(args.revision_threshold*10))

    if "amboss" in args.test_json_filename:
        suffix = suffix +"_"+args.test_json_filename

    module_list = args.reasoning_modules.split(",")
    # if args.mcts_num_last_votes != 32:
    suffix = suffix + "_v"+str(args.num_last_votes)
        # args.mcts_num_last_votes = 32
    if args.num_child_create != 1:
        suffix = suffix + "_c"+str(args.num_child_create)
    if not args.disable_a1:
        suffix = suffix + "_a1"
    if not args.disable_a3:
        suffix = suffix + "_a3"
    if not args.disable_a4:
        suffix = suffix + "_a4"
    if not args.disable_a5:
        suffix = suffix + "_a5"
    if not args.disable_a6:
        suffix = suffix + "_a6"
    if not args.disable_a7:
        suffix = suffix + "_a7"
    if not args.disable_a8:
        suffix = suffix + "_a8"

    args.save_path = "save/" + args.model_ckpt.split("/")[-1] + "_" + args.dataset_name + "_"  + suffix + ".json"
    print("save generation to ", args.save_path)
    main(args)
    import ipdb; ipdb.set_trace()


# system2_info = {}
                # top_score = -1
                # top_freq = -1
                # top_choice = ""
                # freq_choice = ""
                # for node in all_solution_nodes:
                #     if node.module_id == 1:
                #         sys1_choice = node.choice
                #         if node.choice == gold_answer:
                #             num_correct_system1 +=1
                #         continue
                #     choice = node.choice
                #     score = node.node_value
                #     if choice not in system2_info.keys():
                #         system2_info[choice] = [score]
                #     else:
                #         system2_info[choice].append(score)
                #     if sum(system2_info[choice]) > top_score:
                #         top_score = sum(system2_info[choice])
                #         top_choice = choice
                #     if len(system2_info[choice]) > top_freq:
                #         top_freq = len(system2_info[choice])
                #         freq_choice = choice
                # if top_choice == gold_answer:
                #     num_correct_system2_top += 1
                # if freq_choice == gold_answer:
                #     num_correct_system2_freq += 1
                # if top_choice != freq_choice:
                #     import ipdb; ipdb.set_trace()

# sys1_accuracy = num_correct_system1 / num_tested
# sys2_top_accuracy = num_correct_system2_top / num_tested
# sys2_freq_accracy = num_correct_system2_freq / num_tested

# print("System 1 accuracy: ", sys1_accuracy, "System 2 accuracy: ", sys2_top_accuracy)
# if top_choice != gold_answer and sys1_choice == gold_answer:
#     print("check index: ", num_tested-1)
# import ipdb; ipdb.set_trace()
# print(module_acc)

# save_obj['num_system1_correct'] = num_correct_system1
# save_obj['num_system2_top'] = num_correct_system2_top
# save_obj['num_system2_freq'] = num_correct_system2_freq

# save_obj['system2_info'] = system2_info
# patterns = extract_solution_patterns(all_solution_nodes, gold_answer)
# save_obj["correct_patterns"] = patterns

# num_correct_system2_top = save_generations[-1]['num_system2_top']
# num_correct_system2_freq = save_generations[-1]['num_system2_freq']
# num_correct_system1 = save_generations[-1]['num_system1_correct']

# num_correct_system1 = 0
# num_correct_system2_freq = 0
# num_correct_system2_top = 0