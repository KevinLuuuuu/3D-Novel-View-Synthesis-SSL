#!/bin/bash
python3 p1_inference.py --input_dir "${1}" --output_dir "${2}" --config "temp3/dvgo_hotdog/config.py" --render_test --ft_path "temp3/dvgo_hotdog/fine_last.tar"
# TODO - run your inference Python3 code