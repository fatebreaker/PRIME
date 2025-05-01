export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=4,5
export VLLM_WORKER_MULTIPROC_METHOD=spawn
python3 run_src/do_prime.py \
    --dataset_name MedMCQA \
    --test_json_filename test_all \
    --model_ckpt /data/experiment_data_gamma/hieutran/Llama-3.3-70B-Instruct \
    --note default \
    --tensor_parallel_size 2 \
    --num_rollouts 1 \
    --mode run \
    --disable_a5 \
    --disable_a8 \
    --disable_a1 \
    --disable_a3 \
    --disable_a4 \
    --disable_a7 \
    --disable_a6 \
    --mcts_num_last_votes 32 \
    --num_child_create 1 \
    --enable_revision true \
    --revision_file_path "/home/htran/generation/med_preferences/prime/save/Llama-3.3-70B-Instruct_MedMCQA_ro1_decompose_chat_da_v32.json" \
    --revision_threshold 0.7 \
    --num_retrieval 1 \
    --retrieval_corpus "medcorp" \
    --enable_system2_thinking true \
    --enable_chat_template true \
    --enable_reading_documents true \
    --enable_hypothesis_generation true \
#        --enable_direct_answer true \
#    --enable_question_decomposition true \
#    --enable_direct_answer true \
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
