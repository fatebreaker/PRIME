import json

import re
import json

def repair_json_list_file(filepath, output_path=None):
    output_path = output_path or filepath.replace(".json", "_repaired.json")

    with open(filepath, "r") as f:
        raw = f.read()

    # Match entries that begin with "question": and end with "freq_correct_modules": ...}
    pattern = re.compile(r'\{[^{}]*"question"\s*:\s*.*?"freq_correct_modules"\s*:\s*\{.*?\}\s*\}', re.DOTALL)
    matches = pattern.findall(raw)

    if len(matches) > 1:
        matches = matches[:-1]  # Drop the last match (assume incomplete)

    recovered = []
    for i, item in enumerate(matches):
        try:
            recovered.append(json.loads(item))
        except json.JSONDecodeError:
            print(f"❌ Skipping corrupted entry at index {i}")

    with open(output_path, "w") as f:
        json.dump(recovered, f, indent=2)

    print(f"✅ Recovered {len(recovered)} valid entries → saved to: {output_path}")
    return recovered


# Example usage:
# repair_json_list_file("your_broken_file.json")



# Example usage:
repair_json_list_file("/home/htran/generation/med_preferences/prime/save/Llama-3.3-70B-Instruct_MedQA_ro1_r3_textbook_chat_reading_sys2_hypo_v32.json", "/home/htran/generation/med_preferences/prime/save/Llama-3.3-70B-Instruct_MedQA_ro1_r3_textbook_chat_reading_sys2_hypo_v32.json")
