import yaml
from pathlib import Path
from typing import List, Optional


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
    index = {}
    for ln in leaf_nodes:
        path = tuple(ln['key_path'])
        entry = {
            'value': ln['value']
        }
        index.setdefault(path, []).append(entry)
    return index


def _calc_differences(target_leaf_nodes, reference_leaf_nodes):

    diff_count = 0
    for path, target_entries in target_leaf_nodes.items():
        if path not in reference_leaf_nodes:
            diff_count += len(target_entries)
            #print(f"Path {path} not in reference, diff +{len(target_entries)}")
            continue

        
    for path, ref_entries in reference_leaf_nodes.items():
        if path not in target_leaf_nodes:
            diff_count += len(ref_entries)
            #print(f"Path {path} missing in target, diff +{len(ref_entries)}")
            continue

    return diff_count


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

    total_diff_items = 0
    total_reference_items = 0
    result_leaf_nodes = []
    reference_leaf_nodes = []

    for result_dict, reference_dict in zip(result_dict_list, reference_dict_list):
        result_leaf_nodes += _get_leaf_nodes(result_dict)
        reference_leaf_nodes += _get_leaf_nodes(reference_dict)
    
    total_reference_items += len(reference_leaf_nodes)
    
    result_leaf_nodes = _build_leaf_index(result_leaf_nodes)
    reference_leaf_nodes = _build_leaf_index(reference_leaf_nodes)

    total_diff_items += _calc_differences(result_leaf_nodes, reference_leaf_nodes)
    #print(f"Total diff items: {total_diff_items}, Reference items: {total_reference_items}")

    similarity = 1 - (total_diff_items / total_reference_items) if total_reference_items > 0 else 1.0 
    return similarity