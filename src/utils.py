import itertools
import json
import numpy as np
import sys


# --- JSON ---
def json_load(path, report_content=None):
    with open(path, 'r') as f:
        data_json = json.load(f)
    if report_content is not None:
        print_report('LOADED', '%s from %s' % (report_content, path))
    return data_json

class MyJSONEncoder(json.JSONEncoder):
    # Required in order to be able to serialize NumPy numeric types
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(MyJSONEncoder, self).default(obj)

def json_dump(var, path, indent=4, encoder=None):
    if encoder is None:
        encoder = MyJSONEncoder
    with open(path, 'w') as f:
        json.dump(var, f, indent=indent, sort_keys=True, separators=(',', ': '), cls=encoder)


# --- REPORTING ---
def print_progress(string):
    sys.stdout.write('\r%s' % (string))
    sys.stdout.flush()


# --- LIST MANIPULATION ---
def dedup_consecutive(arr):
    return [e for e, _ in itertools.groupby(arr)]
