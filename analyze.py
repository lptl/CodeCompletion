import os
import pickle
from typing import List
from collections import Counter

from tqdm import tqdm
from nltk.translate.bleu_score import sentence_bleu
import numpy as np

from predict import get_prediction, create_model
from tools import prepare_dataset


def get_dataset_predictions(dataset: List[str], groundtruth) -> List[str]:
    """get the predictions of the dataset"""
    predictions = []
    tokenizer, model = create_model()
    for input_text in tqdm(dataset):
        prediction = ""
        retry_times = 0
        while retry_times <= 5 and prediction.strip() == "":
            prediction = get_prediction(
                input_text, tokenizer, model, int(len(groundtruth[0]) * 2)
            )
            middle_index = prediction.find("<fim_middle>")
            eos_index = prediction.find("<|endoftext|>")
            if eos_index == -1:
                prediction = prediction[middle_index + 12]
            else:
                prediction = prediction[middle_index + 12 : eos_index]
            retry_times += 1
        predictions.append(prediction)
    return predictions


def token_accuracy(predicted: str, reference: str) -> float:
    """Calculate token-level accuracy between the predicted and reference tokens."""
    correct = sum(p == r for p, r in zip(predicted, reference))
    return correct / len(reference) if reference else 0.0


def exact_match(predicted: str, reference: str) -> bool:
    """Check if the predicted code exactly matches the reference code."""
    return predicted.strip() == reference.strip()


def edit_distance(predicted: str, reference: str) -> int:
    """Calculate the edit  distance between two code strings."""
    dp = np.zeros((len(reference) + 1, len(predicted) + 1), dtype=int)
    for i in range(len(reference) + 1):
        dp[i][0] = i
    for j in range(len(predicted) + 1):
        dp[0][j] = j

    for i in range(1, len(reference) + 1):
        for j in range(1, len(predicted) + 1):
            if reference[i - 1] == predicted[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])

    return dp[len(reference)][len(predicted)]


def bleu_score(predicted: str, reference: str) -> float:
    """Calculate BLEU score between predicted and reference code."""
    reference_tokens = reference.split()
    predicted_tokens = predicted.split()
    return sentence_bleu([reference_tokens], predicted_tokens)


def compilation_rate(predictions: List[str]) -> float:
    """Calculate the rate at which predicted code compiles successfully."""
    compilable = 0
    for code in predictions:
        try:
            exec(code, {})
            compilable += 1
        except Exception:
            pass
    return compilable / len(predictions) if predictions else 0.0


def get_character_ngrams(text: str, n: int) -> List[str]:
    """Generate n-grams for a given text at the character level."""
    return [text[i : i + n] for i in range(len(text) - n + 1)]


def chrf(predicted: str, reference: str, n: int = 6, beta: float = 2.0) -> float:
    """calculate the chrf score"""
    precision_sum, recall_sum, total_pred_ngrams, total_ref_ngrams = 0.0, 0.0, 0, 0

    for i in range(1, n + 1):
        pred_ngrams = get_character_ngrams(predicted, i)
        ref_ngrams = get_character_ngrams(reference, i)

        pred_counter = Counter(pred_ngrams)
        ref_counter = Counter(ref_ngrams)

        matches = sum((pred_counter & ref_counter).values())

        precision = matches / len(pred_ngrams) if pred_ngrams else 0.0
        recall = matches / len(ref_ngrams) if ref_ngrams else 0.0

        precision_sum += precision
        recall_sum += recall
        total_pred_ngrams += len(pred_ngrams)
        total_ref_ngrams += len(ref_ngrams)

    avg_precision = precision_sum / n
    avg_recall = recall_sum / n

    if avg_precision + avg_recall == 0:
        return 0.0

    beta_squared = beta**2
    chrf_score = (
        (1 + beta_squared)
        * avg_precision
        * avg_recall
        / (beta_squared * avg_precision + avg_recall)
    )

    return chrf_score


def prepare_analysis():
    """produce analysis metrics"""
    if not os.path.exists("predictions.pkl") or not os.path.exists("groundtruth.pkl"):
        dataset, groundtruth = prepare_dataset(directory="/Users/k/Documents/Leetcode")
        predictions = get_dataset_predictions(dataset, groundtruth)
        with open("predictions.pkl", "wb") as f:
            pickle.dump(predictions, f)
        with open("groundtruth.pkl", "wb") as f:
            pickle.dump(groundtruth, f)
        with open("dataset.pkl", "wb") as f:
            pickle.dump(dataset, f)
    else:
        with open("predictions.pkl", "rb") as f:
            predictions = pickle.load(f)
        with open("groundtruth.pkl", "rb") as f:
            groundtruth = pickle.load(f)
        with open("dataset.pkl", "rb") as f:
            dataset = pickle.load(f)

    metrics = {
        "chrf": 0,
        "compilation_rate": 0,
        "bleu_score": 0,
        "edit_distance": 0,
        "exact_match": 0,
        "token_accuracy": 0,
    }
    for i in range(len(predictions)):
        prediction = predictions[i].strip()
        gt = groundtruth[i].strip()
        metrics["chrf"] += chrf(prediction, gt)
        metrics["compilation_rate"] += compilation_rate(prediction)
        metrics["bleu_score"] += bleu_score(prediction, gt)
        metrics["edit_distance"] += edit_distance(prediction, gt)
        metrics["exact_match"] += exact_match(prediction, gt)
        metrics["token_accuracy"] += token_accuracy(prediction, gt)

    for key in metrics:
        metrics[key] /= 50
    return metrics


if __name__ == "__main__":
    print(prepare_analysis())
