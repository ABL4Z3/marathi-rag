from __future__ import annotations

import argparse
from dataclasses import dataclass

from app import answer_question, hybrid_retrieve_documents, load_vector_store


@dataclass(frozen=True)
class EvalCase:
    name: str
    question: str
    required_terms: tuple[str, ...]
    expected_pages: tuple[int, ...]
    forbidden_terms: tuple[str, ...] = ()


EVAL_CASES = [
    EvalCase(
        name="surat_yes_exam_centre",
        question="Is Surat an exam centre in Gujarat?",
        required_terms=("gujarat", "surat", "ahmedabad-gandhinagar", "vadodara"),
        expected_pages=(19,),
    ),
    EvalCase(
        name="surat_negated_question",
        question="Is Surat not the exam centre?",
        required_terms=("gujarat", "surat"),
        expected_pages=(19,),
    ),
    EvalCase(
        name="gujarat_exam_centres",
        question="What are the online examination centres in Gujarat?",
        required_terms=("ahmedabad-gandhinagar", "rajkot", "vadodara", "surat"),
        expected_pages=(19,),
    ),
    EvalCase(
        name="vadodara_exam_centre",
        question="Is Vadodara listed as an online examination centre?",
        required_terms=("vadodara", "surat"),
        expected_pages=(19,),
    ),
    EvalCase(
        name="gujarat_vacancy_cities",
        question="Which Gujarat cities have Junior Assistant vacancies?",
        required_terms=("gujarat", "ahmedabad", "rajkot", "number of vacancies"),
        expected_pages=(1,),
        forbidden_terms=("surat", "vadodara"),
    ),
]


def flatten_docs(docs) -> str:
    return "\n".join(doc.page_content.lower() for doc in docs)


def evaluate_retrieval(final_k: int) -> tuple[int, int]:
    vector_store = load_vector_store()
    if vector_store is None:
        raise RuntimeError("No FAISS index found. Build the knowledge base first.")

    passed = 0
    for case in EVAL_CASES:
        docs = hybrid_retrieve_documents(vector_store, case.question, final_k=final_k)
        text = flatten_docs(docs)
        pages = {int(doc.metadata.get("page", -1)) for doc in docs}
        has_terms = all(term.lower() in text for term in case.required_terms)
        has_page = any(page in pages for page in case.expected_pages)
        avoids_forbidden = all(term.lower() not in text for term in case.forbidden_terms)
        ok = has_terms and has_page and avoids_forbidden
        passed += int(ok)
        top_sources = ", ".join(
            f"p.{doc.metadata.get('page')}#c{doc.metadata.get('chunk_index')}" for doc in docs[:3]
        )
        print(f"{'PASS' if ok else 'FAIL'} retrieval {case.name}: top={top_sources}")
    return passed, len(EVAL_CASES)


def evaluate_answers(final_k: int) -> tuple[int, int]:
    vector_store = load_vector_store()
    if vector_store is None:
        raise RuntimeError("No FAISS index found. Build the knowledge base first.")

    passed = 0
    for case in EVAL_CASES:
        answer, _docs = answer_question(vector_store, case.question, [])
        answer_lc = answer.lower()
        has_terms = all(term.lower() in answer_lc for term in case.required_terms)
        avoids_forbidden = all(term.lower() not in answer_lc for term in case.forbidden_terms)
        ok = has_terms and avoids_forbidden
        passed += int(ok)
        compact_answer = " ".join(answer.split())
        if len(compact_answer) > 240:
            compact_answer = compact_answer[:237] + "..."
        print(f"{'PASS' if ok else 'FAIL'} answer {case.name}: {compact_answer}")
    return passed, len(EVAL_CASES)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate RAG retrieval and optional generated answers.")
    parser.add_argument("--answers", action="store_true", help="Also call the LLM/fallback answer path.")
    parser.add_argument("--final-k", type=int, default=8, help="Number of final chunks used for context.")
    args = parser.parse_args()

    retrieval_passed, retrieval_total = evaluate_retrieval(args.final_k)
    print(f"retrieval_accuracy={retrieval_passed / retrieval_total:.3f} ({retrieval_passed}/{retrieval_total})")

    if args.answers:
        answer_passed, answer_total = evaluate_answers(args.final_k)
        print(f"answer_term_accuracy={answer_passed / answer_total:.3f} ({answer_passed}/{answer_total})")


if __name__ == "__main__":
    main()
