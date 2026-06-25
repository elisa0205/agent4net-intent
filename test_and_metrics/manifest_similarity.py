from pathlib import Path
from metrics import bleu, codeBleu, edit_distance, exact_match, kv_match, kv_wildcard, label_match
from collections import defaultdict
import yaml

GENERATED_PATH = Path("results/llama-4-maverick-17b-128e-instruct-fp8/temp_0.7/stateless-app/gpt-oss-120b_1_attempt_1.yaml")
REFERENCE_PATH = Path("configuration_examples/stateless-app/complete.yaml")

'''
RESULTS:
  BLEU score: 0.8635
  codeBLEU score: 0.6934
  Edit Distance score: 0.6222
  Exact Match score: False
  Label Match score: 0.9655
  KV Match score: 0.6552
  KV Wildcard score: 0.8966
'''

def load_documents(path: str) -> str:
    text = path.read_text(encoding="utf-8")
    return [doc for doc in yaml.safe_load_all(text) if doc is not None]

def group_by_kind(documents):
    grouped = defaultdict(str)

    for doc in documents:
        kind = "Unknown"
        if isinstance(doc, dict):
            kind = doc.get("kind", "Unknown")

        doc_text = yaml.safe_dump(doc, sort_keys=False, allow_unicode=True).strip()
        if grouped[kind]:
            grouped[kind] += "\n---\n" + doc_text
        else:
            grouped[kind] = doc_text

    return grouped


if __name__ == "__main__":
    
    generated_documents = group_by_kind(load_documents(GENERATED_PATH))
    reference_documents = group_by_kind(load_documents(REFERENCE_PATH))

    all_kinds = sorted(set(generated_documents) )
    num_kinds = len(all_kinds)

    bleu_score = 0.0
    code_bleu_score = 0.0
    edit_distance_score = 0.0
    exact_match_score = True
    label_match_score = 0.0
    kv_match_score = 0.0
    kv_wildcard_score = 0.0

    for kind in all_kinds:

      generated_group = generated_documents.get(kind, "")
      #print(generated_group)
      #print("---" * 20)
      reference_group = reference_documents.get(kind, "")
      #print(reference_group)
      if (reference_group == ""):
          num_kinds -= 1
          continue
      
      bleu_score += bleu.test(generated_group, reference_group)
      code_bleu_score += codeBleu.test(generated_group, reference_group)
      edit_distance_score += edit_distance.test(generated_group, reference_group)

      if not exact_match.test(generated_group, reference_group):
          exact_match_score = False

      label_match_score += label_match.test(generated_group, reference_group)
      kv_match_score += kv_match.test(generated_group, reference_group)
      kv_wildcard_score += kv_wildcard.test(generated_group, reference_group)

      #print(bleu_score, code_bleu_score, edit_distance_score, exact_match_score, label_match_score, kv_match_score, kv_wildcard_score)


    # Total scores across all kinds
    print(f"BLEU score: {bleu_score/num_kinds:.4f}")
    print(f"CodeBLEU score: {code_bleu_score/num_kinds:.4f}")
    print(f"Edit Distance score: {edit_distance_score/num_kinds:.4f}")
    print(f"Exact Match score: {exact_match_score}")
    print(f"Label Match score: {label_match_score/num_kinds:.4f}")
    print(f"KV Match score: {kv_match_score/num_kinds:.4f}")
    print(f"KV Wildcard score: {kv_wildcard_score/num_kinds:.4f}")



  