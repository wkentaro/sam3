import argparse

import onnx
import onnx_graphsurgeon

parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument("input_onnx", type=str)
args = parser.parse_args()

graph = onnx_graphsurgeon.import_onnx(onnx.load(args.input_onnx))
print(graph)
