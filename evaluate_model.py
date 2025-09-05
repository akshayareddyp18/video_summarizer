import evaluate
from transformers import pipeline

# -----------------------------
# 1. Dummy Test Dataset
# -----------------------------
# Each entry: ground truth transcript, reference summary, QA (question + answer)
test_data = [
    {
        "ground_truth_transcript": "Artificial intelligence is transforming healthcare by enabling faster diagnosis and better patient care.",
        "asr_output": "Artificial intelligence is transforming healthcare by enabling faster diagnosis and better patient care.",  # perfect match for now
        "reference_summary": "AI is improving healthcare with faster diagnosis and better treatment.",
        "qa_context": "Artificial intelligence is transforming healthcare by enabling faster diagnosis and better patient care.",
        "qa_question": "What is AI transforming?",
        "qa_answer": "Healthcare"
    },
    {
        "ground_truth_transcript": "Climate change is causing rising sea levels and extreme weather patterns around the world.",
        "asr_output": "Climate change is causing rising sea levels and bad weather events globally.",  # slightly different
        "reference_summary": "Climate change leads to higher seas and extreme weather worldwide.",
        "qa_context": "Climate change is causing rising sea levels and extreme weather patterns around the world.",
        "qa_question": "What does climate change cause?",
        "qa_answer": "Rising sea levels and extreme weather"
    }
]

# -----------------------------
# 2. Load Models
# -----------------------------
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
qa_model = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")

# Metrics
wer_metric = evaluate.load("wer")
rouge_metric = evaluate.load("rouge")


# -----------------------------
# 3. Evaluation Functions
# -----------------------------
def evaluate_asr(data):
    refs = [item["ground_truth_transcript"] for item in data]
    hyps = [item["asr_output"] for item in data]
    wer = wer_metric.compute(references=refs, predictions=hyps)
    return wer


def evaluate_summarization(data):
    refs = [item["reference_summary"] for item in data]
    preds = []
    for item in data:
        summary = summarizer(item["asr_output"], max_length=30, min_length=5, do_sample=False)[0]["summary_text"]
        preds.append(summary)
    rouge = rouge_metric.compute(references=refs, predictions=preds)
    return rouge, preds


def evaluate_qa(data):
    em_scores, f1_scores = [], []
    for item in data:
        result = qa_model(question=item["qa_question"], context=item["qa_context"])
        pred_answer = result["answer"].strip().lower()
        true_answer = item["qa_answer"].strip().lower()

        # Exact match
        em = int(pred_answer == true_answer)
        em_scores.append(em)

        # F1 score (token overlap)
        pred_tokens = set(pred_answer.split())
        true_tokens = set(true_answer.split())
        common = pred_tokens.intersection(true_tokens)
        if len(common) == 0:
            f1 = 0.0
        else:
            precision = len(common) / len(pred_tokens)
            recall = len(common) / len(true_tokens)
            f1 = 2 * (precision * recall) / (precision + recall)
        f1_scores.append(f1)

    return {
        "Exact Match": sum(em_scores) / len(em_scores),
        "F1": sum(f1_scores) / len(f1_scores)
    }


# -----------------------------
# 4. Run Evaluation
# -----------------------------
if __name__ == "__main__":
    print("\n🔹 Evaluating ASR (Speech-to-Text)...")
    wer = evaluate_asr(test_data)
    print("WER:", wer)

    print("\n🔹 Evaluating Summarization...")
    rouge, pred_summaries = evaluate_summarization(test_data)
    print("ROUGE:", rouge)
    for i, summary in enumerate(pred_summaries):
        print(f" Predicted Summary {i+1}: {summary}")

    print("\n🔹 Evaluating QA...")
    qa_results = evaluate_qa(test_data)
    print("QA Results:", qa_results)
