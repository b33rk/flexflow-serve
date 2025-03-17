import os
import glob
import pytest
import json

OUTPUT_DIR = os.path.join("inference", "output")

def compare_output_tokens(file1, file2):
    """
    Open two JSON files (each containing a list of dictionaries), check that they have the same number
    of dictionaries, sort them by 'req_idx', and then for each matching req_idx, compare the first 50
    output_tokens (or all tokens if fewer than 50). The output_tokens are stored as a comma-separated
    string of integers under the "output_tokens" key.
    """
    # Helper function to convert a comma-separated string of integers into a list of ints.
    def parse_tokens(token_str):
        return [int(tok.strip()) for tok in token_str.split(',') if tok.strip()]
    
    # Load both JSON files.
    with open(file1, 'r') as f1, open(file2, 'r') as f2:
        data1 = json.load(f1)
        data2 = json.load(f2)
    
    # Check that both files have the same number of dictionaries.
    if len(data1) != len(data2):
        raise ValueError("Error: Files do not have the same number of dictionaries.")
    
    # Sort both lists by the 'req_idx' key.
    data1_sorted = sorted(data1, key=lambda d: d['req_idx'])
    data2_sorted = sorted(data2, key=lambda d: d['req_idx'])
    
    # Compare each pair of dictionaries.
    for d1, d2 in zip(data1_sorted, data2_sorted):
        req_idx1 = d1.get('req_idx')
        req_idx2 = d2.get('req_idx')
        
        # Verify that req_idx values match.
        if req_idx1 != req_idx2:
            raise ValueError(f"Mismatch in req_idx: {req_idx1} vs {req_idx2}")
        
        # Parse the output tokens from the comma-separated strings.
        tokens1 = parse_tokens(d1.get('output_tokens', ''))
        tokens2 = parse_tokens(d2.get('output_tokens', ''))
        
        # Determine the number of tokens to compare.
        num_to_compare = min(30, len(tokens1), len(tokens2))
        if tokens1[:num_to_compare] != tokens2[:num_to_compare]:
            raise ValueError(f"Output tokens mismatch for req_idx {req_idx1} at idx {num_to_compare}/{len(tokens1)}:")


def group_model_files(prefix):
    """
    Returns a dict of the form: {model_name: [list of files]} for files matching:
       [prefix]-python-<model_name>-half_prec*
    """
    pattern = os.path.join(OUTPUT_DIR, f"{prefix}-python-*-half_prec*")
    files = glob.glob(pattern)
    grouped = {}
    for fpath in files:
        basename = os.path.basename(fpath)
        # Example: spec_infer-python-opt-6.7b-half_prec_1_tp_4_pp.json
        remainder = basename[len(prefix + "-python-"):]
        parts = remainder.split("-half_prec", 1)
        if len(parts) < 2:
            continue
        model_name = parts[0]
        grouped.setdefault(model_name, []).append(fpath)
    return grouped

def collect_file_comparisons():
    """
    Yields tuples (file_a, file_b) for all pairwise comparisons among
    spec_infer or incr_dec files that share a model name,
    plus the comparison with huggingface_<model_name>.json if it exists.
    """
    for prefix in ["spec_infer", "incr_dec"]:
        grouped = group_model_files(prefix)
        for model_name, file_group in grouped.items():
            # Pairwise among spec_infer / incr_dec files
            for i in range(len(file_group)):
                for j in range(i+1, len(file_group)):
                    yield file_group[i], file_group[j]
            # Compare with huggingface_<model_name>.json
            hf_file = os.path.join(OUTPUT_DIR, f"huggingface_{model_name}.json")
            if os.path.exists(hf_file) and file_group:
                yield file_group[0], hf_file


def collect_spec_infer_incr_dec_pairs():
    """
    Yields (spec_file, incr_file) for files that have the same trailing name
    after the prefix spec_infer- / incr_dec-.
    """
    all_files = glob.glob(os.path.join(OUTPUT_DIR, "*.*"))  # .json/.json
    spec_infer = {}
    for f in all_files:
        base = os.path.basename(f)
        if base.startswith("spec_infer-"):
            rest = base[len("spec_infer-"):]
            spec_infer[rest] = f
    for f in all_files:
        base = os.path.basename(f)
        if base.startswith("incr_dec-"):
            rest = base[len("incr_dec-"):]
            if rest in spec_infer:
                yield (spec_infer[rest], f)

@pytest.mark.parametrize("file_a,file_b", collect_file_comparisons(), 
                         ids=lambda f: os.path.basename(f))
def test_output_alignment(file_a, file_b):
    """
    Each file pair is tested and reported separately.
    """
    compare_output_tokens(file_a, file_b)



@pytest.mark.parametrize("spec_file,incr_file", collect_spec_infer_incr_dec_pairs(),
                         ids=lambda f: os.path.basename(f))
def test_decoding_steps(spec_file, incr_file):
    """
    Open two JSON files (each containing a list of dictionaries), check that they have the same number
    of dictionaries, sort them by 'req_idx', and then for each matching req_idx, check that the 
    value of 'num_decoding_steps' in spec_file is <= the corresponding value in incr_file / 1.5 times.
    """
    # Load JSON data from both files.
    with open(spec_file, 'r') as f1, open(incr_file, 'r') as f2:
        spec_data = json.load(f1)
        inc_data = json.load(f2)
    
    # Verify that both files contain the same number of dictionaries.
    if len(spec_data) != len(inc_data):
        print("Error: Files do not have the same number of dictionaries.")
        return

    # Sort both lists by the 'req_idx' key.
    data1_sorted = sorted(spec_data, key=lambda d: d['req_idx'])
    data2_sorted = sorted(inc_data, key=lambda d: d['req_idx'])
    
    # Compare each pair of dictionaries.
    for d1, d2 in zip(data1_sorted, data2_sorted):
        req_idx_spec = d1.get('req_idx')
        req_idx_inc_dec = d2.get('req_idx')
        
        # Ensure the req_idx values match.
        if req_idx_spec != req_idx_inc_dec:
            raise ValueError(f"Mismatch in req_idx: {req_idx_spec} vs {req_idx_inc_dec}")
        
        # Get the num_decoding_steps values.
        steps_spec = d1.get('num_decoding_steps')
        steps_inc_dec = d2.get('num_decoding_steps')
        
        if steps_spec is None or steps_inc_dec is None:
            raise ValueError(f"Missing 'num_decoding_steps' for req_idx {req_idx_spec}")
        
        # Check if steps1 is <= 1.5 times steps2.
        if not (steps_spec <= steps_inc_dec / 1.5):
            raise ValueError(f"req_idx {req_idx_spec}: {steps_spec} speculation steps, which is not <= {steps_inc_dec} / 1.5")
