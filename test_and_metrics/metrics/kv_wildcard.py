import yaml
from pathlib import Path
from typing import List, Optional

# default mask file placed next to this module
WILDCARD_MASK = Path("kv_wcm_mask.yaml")

def append_labels_to_keys(kv_labeled_str):
    output_lines = []
    for line in kv_labeled_str.splitlines():
        if '#' in line and line.rstrip().endswith('*'):
            split = line.split(':', 1)
            output_lines.append(split[0].rstrip() + '*:' + split[1]) # rstrip() in case there are spaces trailing the key
        else:
            output_lines.append(line)
    if kv_labeled_str.endswith('\n'):
        output_lines.append('')
    return '\n'.join(output_lines)


def _pattern_to_parts(pat: str) -> List[str]:
    # remove list wildcards "[*]" and split by dot
    return [seg for seg in pat.replace('[*]', '').split('.') if seg != '']

def _matches_pattern(key_path: List[str], pattern_parts: List[str]) -> bool:
    # pattern matches if it's a prefix of the key_path
    if len(pattern_parts) > len(key_path):
        return False
    return key_path[:len(pattern_parts)] == pattern_parts

def get_leaf_nodes(target_dict, wildcard_patterns: Optional[List[str]] = None):
    # wildcard_patterns: list of string patterns like 'spec.template.spec.containers[*].image'
    compiled_patterns = []
    if wildcard_patterns:
        compiled_patterns = [_pattern_to_parts(p) for p in wildcard_patterns]

    leaf_nodes = []
    for key, value in target_dict.items():
        if isinstance(value, dict):
            sub_tree_leaf_nodes = get_leaf_nodes(value, wildcard_patterns)
            leaf_nodes += [{'key_path': [key.rstrip('*')] + leaf_node['key_path'], 
                            'value': leaf_node['value'],
                            'wildcard': leaf_node['wildcard']} 
                            for leaf_node in sub_tree_leaf_nodes]
        elif isinstance(value, list) and len(value) > 0 and isinstance(value[0], dict):
            # all elements in the list should be dict, otherwise we consider it sematically incorrect
            for item in value:
                sub_tree_leaf_nodes = get_leaf_nodes(item, wildcard_patterns)
                leaf_nodes += [{'key_path': [key.rstrip('*')] + leaf_node['key_path'], 
                                'value': leaf_node['value'],
                                'wildcard': leaf_node['wildcard']} 
                                for leaf_node in sub_tree_leaf_nodes]
        else:
            key_path = [key.rstrip('*')]
            # determine if this leaf should be wildcard based on compiled patterns
            is_wildcard = key.endswith('*')
            if not is_wildcard and compiled_patterns:
                # need full key_path from recursive context; for single-level call this is just the key,
                # but higher contexts prepend segments when they build leaf_nodes
                # to properly evaluate, we will leave the exact matching to the caller,
                # but since callers prepend before returning, we check here only on the local key_path.
                # To ensure accurate matching, callers use the returned 'key_path' and compare externally.
                pass
            leaf_nodes.append({'key_path': key_path, 
                               'value': value, 
                               'wildcard': is_wildcard})
    return leaf_nodes

def _apply_compiled_wildcards(leaf_nodes, compiled_patterns: List[List[str]]):
    # leaf_nodes: list of {'key_path': [...], 'value': ..., 'wildcard': bool}
    # compiled_patterns: list of pattern parts (no [*])
    for ln in leaf_nodes:
        kp = ln['key_path']
        for pat in compiled_patterns:
            if _matches_pattern(kp, pat):
                ln['wildcard'] = True
                break
    return leaf_nodes

def calc_intersection(target_leaf_nodes, reference_leaf_nodes):
    intersection = 0
    for reference_leaf_node in reference_leaf_nodes:
        for target_leaf_node in target_leaf_nodes:
            if target_leaf_node['key_path'] == reference_leaf_node['key_path']:
                if reference_leaf_node['wildcard'] \
                or target_leaf_node['value'] == reference_leaf_node['value']:
                    intersection += 1
                    break
    return intersection

def _load_mask(mask_path: Optional[str]) -> List[str]:
    # returns wildcard_paths (list of strings). If mask missing or invalid, return empty list.
    try:
        if mask_path:
            mask_file = Path(mask_path)
        else:
            mask_file = Path(__file__).parent / WILDCARD_MASK
        if not mask_file.exists():
            return []
        raw = yaml.safe_load(mask_file.read_text())
        if not raw:
            return []
        return raw.get('wildcard_paths', []) or []
    except Exception:
        return []

def test(result_str="", reference_str="", mask_path: Optional[str] = None):
    """
    result_str: generated YAML(s)
    reference_str: clean reference YAML(s) (no annotations)
    mask_path: optional path to mask YAML (if None, module default is used)
    """
    try:
        result_dict_list = list(yaml.safe_load_all(result_str))
    except Exception:
        return False

    try:
        reference_dict_list = list(yaml.safe_load_all(reference_str))
        assert len(reference_dict_list) > 0, 'reference_dict_list should not be empty'
    except Exception as e:
        raise Exception(f"Error occurred while loading [reference_str]: {str(e)}") from e
    
    # pad lists to same length
    diff_len = len(result_dict_list) - len(reference_dict_list)
    if diff_len > 0:
        reference_dict_list += [{} for _ in range(diff_len)]
    elif diff_len < 0:
        result_dict_list += [{} for _ in range(abs(diff_len))]

    # load wildcard patterns from mask
    wildcard_patterns = _load_mask(mask_path)
    compiled_patterns = [_pattern_to_parts(p) for p in wildcard_patterns]

    total_intersection = 0
    total_union = 0
    for result_dict, reference_dict in zip(result_dict_list, reference_dict_list):
        # get raw leaf nodes (their key_path values are relative; we need full paths)
        # to build full key paths we reconstruct by using get_leaf_nodes and then
        # prepend ancestors as the function already does; after that apply compiled patterns
        result_leaf_nodes = []
        reference_leaf_nodes = []

        # get_leaf_nodes returns leaf nodes with relative key_path segments built bottom-up,
        # so calling it on the top-level dict yields full paths.
        result_leaf_nodes = get_leaf_nodes(result_dict, wildcard_patterns)
        reference_leaf_nodes = get_leaf_nodes(reference_dict, wildcard_patterns)

        # apply mask patterns to reference_leaf_nodes (we want wildcard semantics defined by mask)
        if compiled_patterns:
            reference_leaf_nodes = _apply_compiled_wildcards(reference_leaf_nodes, compiled_patterns)

        intersection = calc_intersection(result_leaf_nodes, reference_leaf_nodes)
        union = len(result_leaf_nodes) + len(reference_leaf_nodes) - intersection
        total_intersection += intersection
        total_union += union

    if total_union == 0:
        return 1.0
    return total_intersection / total_union