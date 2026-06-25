import yaml
from pathlib import Path
from typing import List, Optional

# default mask file placed next to this module
WILDCARD_MASK = Path("kv_wcm_mask.yaml")


def _pattern_to_parts(pat: str) -> List[str]:
    # remove list wildcards "[*]" and split by dot
    return [seg for seg in pat.replace('[*]', '').split('.') if seg != '']

def _matches_pattern(key_path: List[str], pattern_parts: List[str]) -> bool:
    # pattern matches if it's a prefix of the key_path
    if len(pattern_parts) > len(key_path):
        return False
    return key_path[:len(pattern_parts)] == pattern_parts

def _get_leaf_nodes(target_dict):
    
    leaf_nodes = []
    for key, value in target_dict.items():
        if isinstance(value, dict):
            sub_tree_leaf_nodes = _get_leaf_nodes(value)
            leaf_nodes += [{'key_path': [key] + leaf_node['key_path'], 
                            'value': leaf_node['value']} 
                            for leaf_node in sub_tree_leaf_nodes]
        elif isinstance(value, list) and len(value) > 0 and isinstance(value[0], dict):
            for item in value:
                sub_tree_leaf_nodes = _get_leaf_nodes(item)
                leaf_nodes += [{'key_path': [key] + leaf_node['key_path'], 
                                'value': leaf_node['value']}
                                for leaf_node in sub_tree_leaf_nodes]
        else:
            leaf_nodes.append({'key_path': [key], 
                               'value': value})

    return leaf_nodes

def _build_leaf_index(leaf_nodes):
    # make the list hashable and easily comparable
    # output: {('spec','replicas'): [{'value': 2, 'wildcard': False}, ...], ...}
    index = {}
    for ln in leaf_nodes:
        path = tuple(ln['key_path'])
        entry = {
            'value': ln['value'],
            'wildcard': ln.get('wildcard', False)
        }
        index.setdefault(path, []).append(entry)
    return index

def _apply_compiled_wildcards(leaf_nodes, compiled_patterns: List[List[str]]):
    # leaf_nodes: list of {'key_path': [...], 'value': ...}
    # to all, if the leaf node's key_path matches any of the compiled wildcard patterns, mark it with 'wildcard': True othersize 'wildcard': False
    for ln in leaf_nodes:
        ln['wildcard'] = False
        kp = ln['key_path']
        for pat in compiled_patterns:
            if _matches_pattern(kp, pat):
                ln['wildcard'] = True
                break
    return _build_leaf_index(leaf_nodes)

def _calc_differences(target_leaf_nodes, reference_leaf_nodes):

    # Asymmetric comparison
    diff_count = 0
    for path, target_entries in target_leaf_nodes.items():
        if path not in reference_leaf_nodes:
            if target_entries[0]['wildcard']:
                continue
            diff_count += len(target_entries)
            #print(f"Path {path} not in reference, diff +{len(target_entries)}")
            continue

        ref_entries = reference_leaf_nodes[path]

        if ref_entries[0]['wildcard']:
            continue

        for tgt_entry in target_entries:
            if tgt_entry['value'] not in [ref_entry['value'] for ref_entry in ref_entries]:
                diff_count += 1
                #print(f"Value mismatch at path {path}, target value: {tgt_entry['value']}, reference values: {[ref_entry['value'] for ref_entry in ref_entries]}, diff +1")
                break
        

    return diff_count


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

    total_diff_items = 0
    total_reference_items = 0
    result_leaf_nodes = []
    reference_leaf_nodes = []

    for result_dict, reference_dict in zip(result_dict_list, reference_dict_list):
        result_leaf_nodes += _get_leaf_nodes(result_dict)
        reference_leaf_nodes += _get_leaf_nodes(reference_dict)

    total_reference_items += len(reference_leaf_nodes)

    result_leaf_nodes = _apply_compiled_wildcards(result_leaf_nodes, compiled_patterns)
    reference_leaf_nodes = _apply_compiled_wildcards(reference_leaf_nodes, compiled_patterns)

    total_diff_items += _calc_differences(result_leaf_nodes, reference_leaf_nodes)
    #print(f"Total diff items: {total_diff_items}, Reference items: {total_reference_items}")

    similarity = 1 - (total_diff_items / total_reference_items) if total_reference_items > 0 else 1.0 
    return similarity