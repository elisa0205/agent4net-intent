import yaml
import deepdiff


def count_diff_items(diff):
    total = 0
    for value in diff.values():
        if isinstance(value, dict):
            total += len(value)
        elif isinstance(value, (set, list, tuple)):
            total += len(value)
        else:
            total += 1
    return total

def count_leaf_items(ref):
    if isinstance(ref, dict):
        return sum(count_leaf_items(v) for v in ref.values())
    elif isinstance(ref, list):
        return sum(count_leaf_items(i) for i in ref)
    else:
        return 1

def test(result_str="", reference_str=""):
    try:
        yaml_dict_list1 = list(yaml.safe_load_all(result_str))
    except:
        return 0.0
    
    try:
        yaml_dict_list2 = list(yaml.safe_load_all(reference_str))
        assert len(yaml_dict_list2) > 0, 'yaml_dict_list2 should not be empty'
    except:
        print(f"Invalid reference code:\n{reference_str}")
        return 0.0
    
    if len(list(yaml_dict_list1)) != len(list(yaml_dict_list2)):
        print(f"Number of documents differ: {len(list(yaml_dict_list1))} vs {len(list(yaml_dict_list2))}")
        return 0.0

    total_diff_items = 0
    
    for yaml_dict1, yaml_dict2 in zip(yaml_dict_list1, yaml_dict_list2):
        diff = deepdiff.DeepDiff(yaml_dict1, yaml_dict2, ignore_order=True)
        #print(f"Diff: {diff}")
        if len(diff) > 0:
            total_diff_items += count_diff_items(diff)

    if total_diff_items == 0:
        return 1.0

    reference_docs = list(yaml.safe_load_all(reference_str))
    reference_items = sum(count_leaf_items(doc) for doc in reference_docs if doc is not None)
    #print(f"Total diff items: {total_diff_items}, Reference items: {reference_items}")

    # Normalize the similarity score to be between 0 and 1
    similarity = 1 - (total_diff_items / reference_items) if reference_items > 0 else 1.0
    return similarity