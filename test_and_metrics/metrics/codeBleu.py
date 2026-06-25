from pathlib import Path
import re
from typing import Callable, Dict, List, Optional, Tuple, Union


from codebleu import bleu, weighted_ngram_match
from tree_sitter import Language, Parser
import tree_sitter_yaml

PACKAGE_DIR = Path(__file__).parent

_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_.:/-]+|[{}\[\],?:-]")

YAML_KEYWORDS = {
    "apiversion",
    "kind",
    "metadata",
    "spec",
    "name",
    "namespace",
    "labels",
    "annotations",
    "selector",
    "template",
    "containers",
    "initcontainers",
    "image",
    "ports",
    "env",
    "envfrom",
    "volumeMounts",
    "volumes",
    "replicas",
    "matchlabels",
    "type",
    "data",
    "stringdata",
    "persistentvolumeclaim",
    "service",
    "ingress",
    "deployment",
    "statefulset",
    "daemonset",
    "cronjob",
    "configmap",
    "secret",
    "serviceaccountname",
    "resources",
    "requests",
    "limits",
    "hosts",
    "host",
    "path",
    "paths",
    "pathType",
    "rules",
    "tls",
    "backend",
    "jobtemplate",
    "restartpolicy",
    "imagepullpolicy",
    "nodeselector",
    "affinity",
    "tolerations",
    "policytypes",
    "podselector",
    "ingressclassname",
    "true",
    "false",
    "null",
    "~",
    "yes",
    "no",
    "on",
    "off",
    "-",
    ":",
    "{",
    "}",
    "[",
    "]",
    ",",
    "?",
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
    return {
        token: 1.0 if token.rstrip(":") in YAML_KEYWORDS else 0.2
        for token in reference_tokens
    }


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

    ngram_match_score = bleu.corpus_bleu(tokenized_references, tokenized_hypotheses)

    tokenized_references_with_weights = [
        [[reference_tokens, _build_keyword_weights(reference_tokens)] for reference_tokens in reference_group]
        for reference_group in tokenized_references
    ]
    weighted_ngram_match_score = weighted_ngram_match.corpus_bleu(
        tokenized_references_with_weights,
        tokenized_hypotheses,
    )

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


