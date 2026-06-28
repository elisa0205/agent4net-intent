from pathlib import Path
import re
from typing import Callable, Dict, List, Optional, Tuple, Union


from codebleu import bleu, weighted_ngram_match
from tree_sitter import Language, Parser
import tree_sitter_yaml

PACKAGE_DIR = Path(__file__).parent

_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_.:/-]+|[{}\[\],?:-]")

# Critical for the structure → high weight (2.0)
HIGH_PRIORITY = {
    "apiversion", "kind", "metadata", "spec",
    "containers", "initcontainers", "template", "selector",
    "ingress", "egress", "rules", "ports",
    "volumes", "volumemounts", "persistentvolumeclaim",
    "securitycontext", "serviceaccountname", "rbac",
    "nodeselector", "affinity", "tolerations", "nodename",
}

# Important but not critical → medium weight (1.0)
MEDIUM_PRIORITY = {
    "name", "namespace", "labels", "annotations",
    "image", "command", "args", "workingdir",
    "env", "envfrom", "ports", "lifecycle",
    "resources", "requests", "limits",
    "cpu", "memory", "storage",
    "replicas", "minreplicas", "maxreplicas",
    "livenessprobe", "readinessprobe", "startupprobe",
    "httppget", "tcpsocket", "exec",
    "storageclass", "accessmodes", "volumename",
    "persistentvolume", "capacity",
    "servicename", "clusterip", "loadbalancer",
    "targetport", "nodeport", "protocol",
    "host", "hosts", "path", "paths", "pathtype",
    "ingressclassname", "tls",
    "restartpolicy", "imagepullpolicy", "imagepullsecrets",
    "terminationgraceperiodseconds",
    "schedule", "jobtemplate", "completions", "parallelism",
    "backofflimit", "activedeadlineseconds",
    "data", "stringdata", "binarydata",
    "rules", "verbs", "resources", "apiroups",
    "subjects", "roleref",
    "scaledobject", "metrics", "targetvalue",
}

# Generic YAML syntax → low weight (0.5)
# Document structure, not K8s semantics
LOW_PRIORITY = {
    "true", "false", "null", "~", "yes", "no", "on", "off",
    "-", ":", "{", "}", "[", "]", ",", "?", "|", ">",
    "---", "...",
}


def build_yaml_parser() -> Parser:
    language = Language(tree_sitter_yaml.language(), "yaml")
    parser = Parser()
    parser.set_language(language)
    return parser

def _normalize_references(
    references: Union[List[str], List[List[str]]],
) -> List[List[str]]:
    return [[ref.strip() for ref in sample] if isinstance(sample, list) else [sample.strip()] for sample in references]

def _tokenize_yaml(text: str) -> List[str]:
    return [token.lower() for token in _TOKEN_PATTERN.findall(text)]

def _build_keyword_weights(reference_tokens: List[str]) -> Dict[str, float]:
    weights = {}
    for token in reference_tokens:
        token_clean = token.rstrip(":")
        if token_clean in HIGH_PRIORITY:
            weight = 2.0
        elif token_clean in MEDIUM_PRIORITY:
            weight = 1.0
        elif token_clean in LOW_PRIORITY:
            weight = 0.5
        else:
            weight = 0.2
        weights[token] = weight
    return weights

def _get_all_subtrees(root_node) -> List[str]:
    subtrees: List[str] = []
    stack = [root_node]

    while stack:
        node = stack.pop()
        if node.type == "comment":
            continue

        subtrees.append(str(node))

        for child in reversed(node.children):
            if child.type != "comment":
                stack.append(child)

    return subtrees

# Return a signature string for a given node, representing its type and the signatures of its children.
# This function is introduce to match the stucture of the tree ignoring the order of trees in the structure
def _node_signature(node) -> str:
    if node.child_count == 0:
        # Leaf node
        return f"{node.type}:{node.text.decode('utf8').lower()}"
    
    # Internal node: recursion on children
    children_sig = ",".join(
        _node_signature(c) for c in node.children 
        if c.type != "comment"
    )
    return f"{node.type}({children_sig})"

def _get_all_subtrees(root_node) -> List[str]:
    subtrees = []
    stack = [root_node]

    while stack:
        node = stack.pop()
        if node.type == "comment":
            continue

        subtrees.append(_node_signature(node))  

        for child in reversed(node.children):
            if child.type != "comment":
                stack.append(child)
    return subtrees

def _corpus_syntax_match_yaml(
    references: List[List[str]],
    candidates: List[str],
    parser: Parser,
) -> float:
    match_count = 0
    total_count = 0

    for reference_group, candidate in zip(references, candidates):
        candidate_tree = parser.parse(candidate.encode("utf8")).root_node
        candidate_subtrees = set(_get_all_subtrees(candidate_tree))

        for reference in reference_group:
            reference_tree = parser.parse(reference.encode("utf8")).root_node
            reference_subtrees = _get_all_subtrees(reference_tree)

            for sub_tree in reference_subtrees:
                if sub_tree in candidate_subtrees:
                    match_count += 1

            total_count += len(reference_subtrees)

    if total_count == 0:
        return 0.0

    return match_count / total_count

def calc_codebleu_yaml(
    references: Union[List[str], List[List[str]]],
    predictions: List[str],
    weights: Tuple[float, float, float, float] = (0.33, 0.33, 0.34, 0.0),
    tokenizer: Optional[Callable[[str], List[str]]] = None,
) -> Dict[str, float]:
    assert len(references) == len(predictions), "Number of references and predictions should be the same"
    assert len(weights) == 4, "weights should be a tuple of 4 floats"

    normalized_references = _normalize_references(references)
    hypotheses = [prediction.strip() for prediction in predictions]

    if tokenizer is None:
        tokenizer = _tokenize_yaml

    tokenized_hypotheses = [tokenizer(hypothesis) for hypothesis in hypotheses]
    tokenized_references = [[tokenizer(reference) for reference in reference_group] for reference_group in normalized_references]

    # BLEU score
    ngram_match_score = bleu.corpus_bleu(tokenized_references, tokenized_hypotheses)

    # Weighted N-gram Match score
    tokenized_references_with_weights = [
        [[reference_tokens, _build_keyword_weights(reference_tokens)] for reference_tokens in reference_group]
        for reference_group in tokenized_references
    ]
    weighted_ngram_match_score = weighted_ngram_match.corpus_bleu(
        tokenized_references_with_weights,
        tokenized_hypotheses,
    )

    # Syntax Match score - AST 
    parser = build_yaml_parser()
    syntax_match_score = _corpus_syntax_match_yaml(normalized_references, hypotheses, parser)

    dataflow_match_score = 0.0

    alpha, beta, gamma, theta = weights
    code_bleu_score = (
        alpha * ngram_match_score
        + beta * weighted_ngram_match_score
        + gamma * syntax_match_score
        + theta * dataflow_match_score
    )

    return {
        "codebleu": code_bleu_score,
        "ngram_match_score": ngram_match_score,
        "weighted_ngram_match_score": weighted_ngram_match_score,
        "syntax_match_score": syntax_match_score,
        "dataflow_match_score": dataflow_match_score,
    }



def test(result_str="", reference_str=""):
    references = [reference_str]
    predictions = [result_str]

    scores = calc_codebleu_yaml(references, predictions)
    return scores["codebleu"]


