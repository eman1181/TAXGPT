print("Starting retrieval evaluation...", flush=True)

import pandas as pd
from pathlib import Path

print("Checking evaluation_questions.csv...", flush=True)

csv_path = Path("evaluation_questions.csv")

if not csv_path.exists():
    print("evaluation_questions.csv not found. Creating sample file...", flush=True)

    sample_data = [
        {
            "id": 1,
            "question": "ما هو المقصود بنسبة 0.0025 المقررة بالمادة 40 من قانون التأمين الصحي الشامل؟",
            "expected_source": "health_contribution",
        },
        {
            "id": 2,
            "question": "ما هي قواعد ضريبة الدخل؟",
            "expected_source": "income_tax",
        },
        {
            "id": 3,
            "question": "ما هي تعليمات سنة 2012؟",
            "expected_source": "instructions_2012",
        },
        {
            "id": 4,
            "question": "ما هي الفتوى الضريبية بخصوص القيمة المضافة؟",
            "expected_source": "fatwa",
        },
    ]

    pd.DataFrame(sample_data).to_csv(
        csv_path,
        index=False,
        encoding="utf-8-sig"
    )

print("Importing retrieve function from GRAD_FINAL_BACKEND.py...", flush=True)

from GRAD_FINAL_BACKEND import retrieve

print("Backend imported successfully.", flush=True)


def evaluate_retrieval(csv_path="evaluation_questions.csv", top_k=5):
    print(f"Reading CSV: {csv_path}", flush=True)

    df = pd.read_csv(csv_path)

    print(f"Loaded {len(df)} questions.", flush=True)

    total = len(df)
    correct_at_1 = 0
    correct_at_k = 0
    results = []

    for index, row in df.iterrows():
        question = row["question"]
        expected_source = row["expected_source"]

        print("\n" + "=" * 80, flush=True)
        print(f"Testing question {index + 1}/{total}", flush=True)
        print("Question:", question, flush=True)
        print("Expected source:", expected_source, flush=True)

        retrieved = retrieve(question, top_k=top_k)
        retrieved_sources = [r["source"] for r in retrieved]

        print("Retrieved sources:", retrieved_sources, flush=True)

        hit_at_1 = retrieved_sources[0] == expected_source
        hit_at_k = expected_source in retrieved_sources

        print("Hit@1:", hit_at_1, flush=True)
        print(f"Hit@{top_k}:", hit_at_k, flush=True)

        if hit_at_1:
            correct_at_1 += 1

        if hit_at_k:
            correct_at_k += 1

        results.append({
            "id": row["id"],
            "question": question,
            "expected_source": expected_source,
            "top_1_source": retrieved_sources[0],
            f"hit_at_{top_k}": hit_at_k,
            "retrieved_sources": " | ".join(retrieved_sources),
        })

    recall_1 = correct_at_1 / total if total else 0
    recall_k = correct_at_k / total if total else 0

    print("\nRetrieval Evaluation Results")
    print("----------------------------")
    print(f"Total questions: {total}")
    print(f"Recall@1: {recall_1:.2%}")
    print(f"Recall@{top_k}: {recall_k:.2%}")

    output_path = "retrieval_evaluation_results.csv"

    pd.DataFrame(results).to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig"
    )

    print(f"\nSaved results to {output_path}", flush=True)


if __name__ == "__main__":
    evaluate_retrieval()