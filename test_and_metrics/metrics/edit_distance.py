from nltk.metrics.distance import edit_distance

# Calculate the Levenshtein edit-distance between two strings
def test(result_str="", reference_str=""):

    edit_dist_score = edit_distance(result_str, reference_str)
    max_len = max(len(result_str), len(reference_str))

    # Calculate text similarity in the range [0, 1], where 1 means identical strings and 0 means completely different strings
    similarity = 1 - (edit_dist_score / max_len) if max_len > 0 else 1
    return similarity



