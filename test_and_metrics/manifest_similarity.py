from pathlib import Path
from collections import defaultdict
from metrics import bleu, codeBleu, edit_distance, exact_match, kv_match, kv_wildcard, key_match

import yaml

GENERATED_ROOT = Path("test1-results")
REFERENCE_ROOT = Path("configuration_examples")

'''
RESULTS:
  BLEU score: 0.8635
  codeBLEU score: 0.6934
  Edit Distance score: 0.6222
  Exact Match score: False
  Key Match score: 0.9655
  KV Match score: 0.6552
  KV Wildcard score: 0.8966
'''

def load_documents(path: Path):
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

def get_reference_path(generated_path: Path) -> Path:
    relative_path = generated_path.parent.name
    return REFERENCE_ROOT / relative_path / "complete.yaml"

def compute_kind_precision_recall(generated_documents: dict, reference_documents: dict) -> dict[str, float]:
    """
    Precision/recall on *kind*
 
    - Precision: how many of the generated kinds were actually expected in the reference
    - Recall: what fraction of the expected kinds in the reference were actually generated
    """
    generated_kinds = set(generated_documents)
    reference_kinds = set(reference_documents)
 
    common_kinds = generated_kinds & reference_kinds
    extra_kinds = sorted(generated_kinds - reference_kinds)   # generated but not expected (false positives)
    missing_kinds = sorted(reference_kinds - generated_kinds) # expected but not generated (false negatives)
 
    precision = len(common_kinds) / len(generated_kinds) if generated_kinds else 0.0
    recall = len(common_kinds) / len(reference_kinds) if reference_kinds else 0.0
 
    return {
        "kind_precision": precision,
        "kind_recall": recall,
        "extra_kinds": extra_kinds,
        "missing_kinds": missing_kinds,
    }


def format_report(
    generated_path: Path,
    reference_path: Path,
    metrics: dict[str, float | bool],
    num_kinds: int,
    kind_metrics: dict,
) -> str:
    lines = [
        f"Generated file: {generated_path}",
        f"Reference file: {reference_path}",
        f"Matched kinds: {num_kinds}",
        f"BLEU: {metrics['bleu_score']:.4f}",
        f"CodeBLEU: {metrics['code_bleu_score']:.4f}",
        f"Edit_Distance: {metrics['edit_distance_score']:.4f}",
        f"Exact_Match: {metrics['exact_match_score']}",
        f"Key_Match: {metrics['key_match_score']:.4f}",
        f"KV_Match: {metrics['kv_match_score']:.4f}",
        f"KV_Wildcard: {metrics['kv_wildcard_score']:.4f}",
        f"Kind_Precision: {kind_metrics['kind_precision']:.4f}",
        f"Kind_Recall: {kind_metrics['kind_recall']:.4f}",
        f"Extra_Kinds: {', '.join(kind_metrics['extra_kinds']) or '-'}",
        f"Missing_Kinds: {', '.join(kind_metrics['missing_kinds']) or '-'}",
        "",
    ]
    return "\n".join(lines)
 


if __name__ == "__main__":
    generated_files = sorted(GENERATED_ROOT.rglob("*.yaml"))

    for generated_path in generated_files:
        reference_path = get_reference_path(generated_path)
        output_path = generated_path.with_suffix(".txt")

        if not reference_path.exists():
            output_path.write_text(
                f"Generated file: {generated_path}\n"
                f"Reference file not found: {reference_path}\n",
                encoding="utf-8",
            )
            continue
 
        generated_documents = group_by_kind(load_documents(generated_path))
        reference_documents = group_by_kind(load_documents(reference_path))
 
        # Precision/recall/F1 sui kind: calcolate sempre, indipendentemente
        # dal fatto che esistano o meno kind in comune.
        kind_metrics = compute_kind_precision_recall(generated_documents, reference_documents)
 
        common_kinds = sorted(set(generated_documents) & set(reference_documents))
        if not common_kinds:
            lines = [
                f"Generated file: {generated_path}",
                f"Reference file: {reference_path}",
                "No common kinds found.",
                "",
                f"Extra_Kinds: {', '.join(kind_metrics['extra_kinds']) or '-'}",
                f"Missing_Kinds: {', '.join(kind_metrics['missing_kinds']) or '-'}",
                "",
            ]
            output_path.write_text("\n".join(lines), encoding="utf-8")
            continue
 
        aggregate_metrics = {
            "bleu_score": 0.0,
            "code_bleu_score": 0.0,
            "edit_distance_score": 0.0,
            "exact_match_score": 1,
            "key_match_score": 0.0,
            "kv_match_score": 0.0,
            "kv_wildcard_score": 0.0}
 
        for kind in common_kinds:
            generated_group = generated_documents.get(kind, "")
            reference_group = reference_documents.get(kind, "")
 
            aggregate_metrics["bleu_score"] += bleu.test(generated_group, reference_group)
            aggregate_metrics["code_bleu_score"] += codeBleu.test(generated_group, reference_group)
            aggregate_metrics["edit_distance_score"] += edit_distance.test(generated_group, reference_group)
 
            if not exact_match.test(generated_group, reference_group):
                aggregate_metrics["exact_match_score"] = 0
 
            aggregate_metrics["key_match_score"] += key_match.test(generated_group, reference_group)
            aggregate_metrics["kv_match_score"] += kv_match.test(generated_group, reference_group)
            aggregate_metrics["kv_wildcard_score"] += kv_wildcard.test(generated_group, reference_group)
 
        num_kinds = len(common_kinds)
 
        averaged_metrics = {
            "bleu_score": aggregate_metrics["bleu_score"] / num_kinds,
            "code_bleu_score": aggregate_metrics["code_bleu_score"] / num_kinds,
            "edit_distance_score": aggregate_metrics["edit_distance_score"] / num_kinds,
            "exact_match_score": aggregate_metrics["exact_match_score"],
            "key_match_score": aggregate_metrics["key_match_score"] / num_kinds,
            "kv_match_score": aggregate_metrics["kv_match_score"] / num_kinds,
            "kv_wildcard_score": aggregate_metrics["kv_wildcard_score"] / num_kinds}
 
        report = format_report(generated_path, reference_path, averaged_metrics, num_kinds, kind_metrics)
        output_path.write_text(report, encoding="utf-8")




  