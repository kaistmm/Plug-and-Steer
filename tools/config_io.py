"""Reading configs back, including the ones the trainer wrote itself.

Why this exists. `train.py` loads configs with ruamel and the trainer dumps the
running config to `exps/<tag>/config.yaml` with ruamel's round-trip dumper. That
dumper emits floats in the YAML 1.2 short form (`1e-05`, no decimal point).
PyYAML implements YAML 1.1, where that token is a *string*, so anything reading a
saved config with `yaml.safe_load` gets `eps='1e-05'` and hands a str to the
model constructor.

Reading with ruamel keeps the round-trip consistent; `coerce_numeric_strings`
additionally repairs configs that were already written by an older PyYAML.
"""

from ruamel.yaml import YAML

_yaml = YAML(typ='safe', pure=True)


def coerce_numeric_strings(node):
    """Recursively turn '1e-05'-style strings back into floats."""
    if isinstance(node, dict):
        return {k: coerce_numeric_strings(v) for k, v in node.items()}
    if isinstance(node, list):
        return [coerce_numeric_strings(v) for v in node]
    if isinstance(node, str):
        try:
            return float(node)
        except ValueError:
            return node
    return node


def load_config(path, coerce=True):
    with open(path, 'r', encoding='utf-8') as f:
        config = _yaml.load(f)
    return coerce_numeric_strings(config) if coerce else config
