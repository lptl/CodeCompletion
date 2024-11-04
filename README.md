# Report

## Requirements

- Generate a dataset of code completion examples from one of your own repositories. You can take a few files from your personal projects.
- Write a script that would split them into three parts simulating the user cursor position: the prefix - code before the cursor, the suffix - code after the cursor, and the middle - code that is missing and we assume should be typed next. You should aim to obtain between 20 and 50 examples.
- Take an open source code completion model, for example tiny_starcoder, or bigger starcoder or codellama variations if you prefer. Run the selected model on the examples from the previous point to obtain its completions.
- Review them manually and analyze the differences with actual missing code from the middle, try to assign labels according to your judgment of whether the proposed code is correct.
- Try to propose some automatic metrics for evaluating the quality of this model. Find which of the proposed metrics correlates better with your own judgment. Try computing at least exact match, chrf, and two or more metrics of your choice.
- Submit the code you wrote in the process, the resulting dataset with your annotations and computed metrics, and a report in any format describing your thought process, findings and learnings.

## Files and Model

`predict.py` contains the prediction functionality. `tools.py` contains the tools used to generate dataset. `analyze.py` contains the functions used to analyze the result. The used model is `tiny_startcoder` with `cpu` inferencing.

The execution commands is as follows:

```bash
poetry install
poetry run python analyze.py
```

The number of samples is 50. The output result is as follows:

```
{'chrf': 0.27838972155589237, 'compilation_rate': 0.16921495513542514, 'bleu_score': 0.07469626926717826, 'edit_distance': np.float64(37.1), 'exact_match': 0.12, 'token_accuracy': 0.23704617997966804}
```

To see the data used:

```python
import pickle

# load the dataset used
with open("dataset.pkl") as f:
    dataset = pickle.load(f)

# load the groundtruth
with open("groundtruth.pkl") as f:
    groundtruth = pickle.load(f)

# load the model prediction
with open("predictions.pkl") as f:
    predictions = pickle.load(f)
```

## Thoughts

1. choose an appropriate folder. Considering the model capatability and the code I am having, the folder I chose contains my leetcode solutions. I filtered python solutions and only used them. However, I think for simple models, the code samples should be explicit and clear, instead of relying too much context.

2. The split dataset part requries me to find out the suitable index location. The situation here is that I try to add more context to the model so that the `suffix` and `prefix` part contains enough content to predict the missing one line of the `mid`. I only wrote a simple parser here, and in actual development, the parser could be much more complicated. The number of files inside my folder is 126. Each file contains 20~100 lines of code.

3. When evaluating the metrics, the exact match would be a good strategy. But multiple correct codes could semantically or grammarly exist multiple different solutions. Thus, it would be better to use another LLM that can understand code to evaluate the equivalence between generated one and the groundtruth one. In this part I used some common string matching algorithm to calculate the distance.

## Learnings

The dataset generation part is quite important. Figuring out the suitable training dataset format is quite critical for the result model performance. Not only the code style issue should be considered, when encountering large projects, how to decide preliminary context would be hard.

Moreover, the evaluation metric is not very easy to make especially when the generated code is long and achieves some complex functions. Manual evaluation would be okay, otherwise, another LLM would be needed to automate the evaluation strategy. One simple strategy for the evaluation could use the execution result of the produced code and compare the result with the original one.
