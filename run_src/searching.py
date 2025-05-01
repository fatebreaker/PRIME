import requests
import json
import re


def extract_between(text: str, start_tag: str, end_tag: str):
    pattern = re.escape(start_tag) + r"(.*?)" + re.escape(end_tag)
    matches = re.findall(pattern, text, flags=re.DOTALL)
    if matches:
        return matches
    return None


BEGIN_SEARCH_QUERY = "<|begin_search_query|>"
END_SEARCH_QUERY = "<|end_search_query|>"
BEGIN_SEARCH_RESULT = "<|begin_search_result|>"
END_SEARCH_RESULT = "<|end_search_result|>"
num_searches = 1

# text = "<|begin_search_query|>causes of flaccid paralysis in infants<|end_search_query|>\n<|begin_search_query|>pathophysiology of antibodies against postsynaptic nicotinic cholinergic ion channels<|end_search_query|>\n<|begin_search_query|>pathophysiology of autoantibodies against presynaptic voltage-gated calcium channels<|end_search_query|><|begin_search_query|>pathophysiology of autoimmune demyelination of peripheral nerves<|end_search_query|><|begin_search_query|>pathophysiology of blockade of presynaptic acetylcholine release at the neuromuscular junction<|end_search_query|><|begin_search_query|>pathophysiology of lower motor neuron destruction in anterior horn<|end_search_query|><|begin_search_query|>causes of bilateral ptosis in infants<|end_search_query|><|begin_search_query|>causes of weak gag reflex and poor sucking in infants<|end_search_query|><|begin_search_query|>causes of constipation and urinary retention in infants<|end_search_query|><|begin_search_query|>descending paralysis clinical features in infants<|end_search_query|>"
# text = "<|begin_search_query|>causes of bilateral sensorineural hearing loss in young adults<|end_search_query|><|begin_search_query|>causes of facial muscle weakness bilaterally<|end_search_query|><|begin_search_query|>causes of unsteady gait with hearing loss<|end_search_query|><|begin_search_query|>common findings in masses at the internal auditory meatus<|end_search_query|><|begin_search_query|>common findings in masses at the cerebellopontine angle<|end_search_query|><|begin_search_query|>skin findings of yellow plaques and papules in systemic diseases<|end_search_query|><|begin_search_query|>embryologic origin of neural tube<|end_search_query|><|begin_search_query|>embryologic origin of surface ectoderm<|end_search_query|><|begin_search_query|>embryologic origin of neural crest<|end_search_query|><|begin_search_query|>embryologic origin of notochord<|end_search_query|><|begin_search_query|>embryologic origin of mesoderm<|end_search_query|>"
text = "<|begin_search_query|>causes of chronic productive cough in adults<|end_search_query|><|begin_search_query|>causes of exertional dyspnea in adults<|end_search_query|><|begin_search_query|>effects of long-term smoking on lung function<|end_search_query|><|begin_search_query|>clinical findings associated with wheezing and rhonchi<|end_search_query|><|begin_search_query|>chronic decrease in pulmonary compliance and clinical features<|end_search_query|><|begin_search_query|>local accumulation of kinins and respiratory symptoms<|end_search_query|><|begin_search_query|>mycobacterial infections affecting pulmonary parenchyma<|end_search_query|><|begin_search_query|>progressive obstruction of expiratory airflow and clinical presentation<|end_search_query|><|begin_search_query|>loss of functional residual capacity and its effects<|end_search_query|>"


queries = extract_between(text, BEGIN_SEARCH_QUERY, END_SEARCH_QUERY)
all_documents = ""
# import ipdb; ipdb.set_trace()
for search_query in queries:
    query = "http://172.16.34.21:6000/api/search?query="+search_query+"&k="+str(num_searches)
    x = requests.get(query)
    jsobj = json.loads(x.text)
    document = ""
    for idx in range(len(jsobj['topk'])):
        document = document + jsobj['topk'][idx]['text']
    print(document)
    # import ipdb; ipdb.set_trace()
