export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=2
export VLLM_WORKER_MULTIPROC_METHOD=spawn
python3 run_src/do_prime.py \
    --dataset_name MedMCQA \
    --test_json_filename test_all \
    --model_ckpt /data/data_user_alpha/public_models/Llama-3.1/Meta-Llama-3.1-8B-Instruct \
    --note default \
    --tensor_parallel_size 1 \
    --num_rollouts 1 \
    --mode run \
    --disable_a5 \
    --disable_a8 \
    --disable_a1 \
    --disable_a3 \
    --disable_a4 \
    --disable_a7 \
    --disable_a6 \
    --mcts_num_last_votes 128 \
    --num_child_create 1 \
    --enable_direct_answer true \
#    --enable_revision true \
#    --revision_file_path "/home/htran/generation/med_preferences/prime/save/Meta-Llama-3.1-8B-Instruct_MedQA_ro1_da_v32.json" \
#    --revision_threshold 0.7 \
#    --num_retrieval 1 \
#    --retrieval_corpus "textbook" \
#    --enable_system2_thinking true \
#    --half_precision \
#    --enable_reading_documents true \
#    --enable_hypothesis_generation true \
#    --num_queries 3 \
#    --enable_reading_documents true \
#    --enable_direct_answer true \
#    --enable_question_decomposition true \
#    --enable_chat_template true \
    #    --enable_system2_thinking true \
#    --start_from 874 \
#    --enable_direct_answer true \
#        --disable_a7 \
#    --disable_a6 \
#    --retrieval_corpus "wikipedia" \
#    --retrieval_threshold 0.5 \
#    --num_queries 5 \
#    --num_retrieval 3 \
#        --disable_a6 \
#    --enable_chat_template
#    --retrieval_corpus "medcorp" \
#    --retrieval_threshold 0.5 \
#    --enable_answer_checking true \
#    --combine_distributions "disagreement" \
#    --search_query_weight 0.5 \
#    --enable_majority true \
#    --combine_distributions "multi" \
#    --combine_distributions "subtract" \
#    --majority_threshold 0.4
#    --disable_a8 \
#    --disable_a6 \
#    --disable_a7
#    --disable_answer_selection \
#    --api llama \
#        --model_ckpt llama31_8b_with_openai_api \
#    --tensor_parallel_size 4
