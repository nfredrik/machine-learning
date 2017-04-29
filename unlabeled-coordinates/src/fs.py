"""File system related utilities."""

import json


def write_json_to(fname: str, data: dict, pretty: bool=False) -> None:
    with open(fname, 'w') as f:
        if pretty:
            json.dump(data, f, indent=4)
        else:
            json.dump(data, f)
