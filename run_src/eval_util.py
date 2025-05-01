import jsonlines
import json


def load_json(filename):
    """
    Load a JSON file given a filename
    If the file doesn't exist, then return an empty dictionary instead
    """
    try:
        with open(filename, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return {}


def load_jsonl(filename):
    file_content = []
    try:
        with jsonlines.open(filename) as reader:
            for obj in reader:
                file_content.append(obj)
            return file_content
    except FileNotFoundError:
        return []


def write_jsonl(data, filepath):
    with open(filepath, 'w') as jsonl_file:
        for line in data:
            jsonl_file.write(json.dumps(line))
            jsonl_file.write('\n')


def jsonl2json(jsonl_filepath, json_filepath):
    # # The path to your .jsonl file
    # jsonl_filepath = 'test.jsonl'

    # # The path where you want to save your .json file
    # json_filepath = 'test.json'

    # Read the JSON lines from the .jsonl file
    with open(jsonl_filepath, 'r') as file:
        jsonl_lines = file.readlines()

    # Initialize the list of instances
    instances = []

    # Process each line
    for line in jsonl_lines:
        # Load the JSON object from the line
        obj = json.loads(line)
        # print(obj.keys())
        # Construct the question string
        # input_string = f"Question: {obj['question']} "
        # for option_key in sorted(obj['options'].keys()):
        #     input_string += f"({option_key}) {obj['options'][option_key]} "

        input_string = "Instruction: {instruction}\n\nQuestion: {question}".format(
            instruction=obj['instruction'],
            question=obj['input']
        )
        # Find the answer index
        # output_string = [key for key, value in obj['options'].items() if value == obj['answer']][0]
        output_string = obj['output']
        # Append the new instance to the list
        instances.append({
            'input': input_string,
            'output': output_string
        })
        # instances.append(obj)
    print(instances[0])
    # Save the instances to the .json file
    with open(json_filepath, 'w') as file:
        json.dump({'type': 'text2text', 'instances': instances}, file, indent=2)


if __name__ == '__main__':
    train_data = load_jsonl('/data/qa_tuning_data/qa_instructions_train.jsonl')
    dev_data = load_jsonl('/data/qa_tuning_data/qa_instructions_dev.jsonl')

    print(len(train_data), len(dev_data))

    # jsonl2json('/data/qa_tuning_data/qa_instructions_train.jsonl', '/data/qa_tuning_data/lmflow_train/qa_instructions_train.json')
    # jsonl2json('/data/qa_tuning_data/qa_instructions_dev.jsonl', '/data/qa_tuning_data/lmflow_eval/qa_instructions_dev.json')