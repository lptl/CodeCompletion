import os
from typing import List
import random


def find_function(file_content: str):
    """find the functions inside the file_content"""
    lines = file_content.split("\n")
    content = []

    start_idx = None
    end_idx = None
    for i, line in enumerate(lines):
        if line.strip().startswith("def"):
            start_idx = i
            def_location = line.find("def")
            for j, l in enumerate(lines[i:]):
                if (
                    l.strip().startswith("return")
                    and l.find("return") - def_location == 4
                ):
                    end_idx = i + j
                    break

        if start_idx is not None and end_idx is not None:
            function_content = "\n".join(lines[start_idx : end_idx + 1])
            content.append(function_content)
    return content


def prepare_dataset(directory: str) -> List[str]:
    """prepare the dataset and related groundtruth"""
    dataset = []
    groundtruth = []

    files = os.listdir(directory)
    random.Random(7).shuffle(files)
    for filename in files:
        if filename.endswith(".py"):
            with open(os.path.join(directory, filename), "r") as file_handler:
                file_content = file_handler.read()
                functions = find_function(file_content)
                if len(functions):
                    # i only process one function one file
                    function = functions[0].split("\n")
                    length = len(function)
                    index = random.randint(0, length - 1)
                    while (
                        function[index].strip() == ""
                        or function[index].strip().startswith("#")
                        or index / length >= 0.85
                        or index / length <= 0.15
                    ):
                        index = random.randint(0, length - 1)
                    prefix = "<fim_prefix>" + "\n".join(function[:index])
                    suffix = "<fim_suffix>" + "\n".join(function[index + 1 :])
                    dataset.append(prefix + suffix + "<fim_middle>")
                    groundtruth.append(function[index])
                    if len(dataset) >= 50:
                        break
    return (dataset, groundtruth)


if __name__ == "__main__":
    dataset, groundtruth = prepare_dataset(directory="/Users/k/Documents/Leetcode")
    print(len(dataset), len(groundtruth))
    print(dataset[13], groundtruth[13])
