import os
import json
import time

TEST_DATA_FOLDER = "data/invoices"


def test_all(process_fun, output_folder):
    for file in os.listdir(TEST_DATA_FOLDER):
        print(f"File: {file}")
        t0 = time.time()
        out = process_fun(os.path.join(TEST_DATA_FOLDER, file))
        if isinstance(out, list):
            if len(out) == 1:
                out = out[0]
                out["time"] = time.time() - t0
        else:
            out["time"] = time.time() - t0
                

        with open(os.path.join(output_folder, os.path.splitext(file)[0] + ".json"), "w") as f:
            json.dump(out, f, indent=2)

        print(out)


def text_to_json(text):
    json_start = text.find("```json\n") + len("```json\n")
    json_end = text.find("```", json_start)
    json_str = text[json_start:json_end].strip()

    out = json.loads(json_str)
    return out