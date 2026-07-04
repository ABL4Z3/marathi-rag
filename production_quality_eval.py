from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from app import (
    answer_question,
    build_chunks,
    build_vector_store,
    extract_docx_documents,
    extract_pdf_documents,
)


@dataclass(frozen=True)
class Case:
    name: str
    question: str
    required: tuple[str, ...] = ()
    forbidden: tuple[str, ...] = ()
    should_refuse: bool = False


CASES = [
    Case(
        "resume_father_typo",
        "what is hte name of anuj father",
        required=("Shri Khem Raj Singh",),
        forbidden=("MosquitoFusion",),
    ),
    Case(
        "resume_address_short",
        "address",
        required=("Badshahpur", "Bijnor", "246747"),
        forbidden=("MosquitoFusion",),
    ),
    Case(
        "resume_phone_short",
        "phone no",
        required=("+91-9528872784",),
        forbidden=("MosquitoFusion",),
    ),
    Case(
        "resume_qualities_typo",
        "what is hte qulaities",
        required=("Quick learner", "Hardworking", "self-motivated"),
        forbidden=("MosquitoFusion",),
    ),
    Case(
        "research_abstract_typo",
        "what is the abstract of mosquitp",
        required=("MosquitoFusion", "1,204", "YOLOv8s", "GIS"),
        forbidden=("Father", "Bijnor"),
    ),
    Case(
        "research_methodology_typo",
        "give me the methodology og that resarch paper",
        required=("Data Preprocessing", "YOLOv8s", "GIS Integration"),
        forbidden=("Father", "Bijnor"),
    ),
    Case(
        "unsupported_resume_salary",
        "what is anuj salary",
        should_refuse=True,
    ),
]


REFUSAL_MARKERS = (
    "could not find",
    "not found",
    "not clearly supported",
    "स्पष्ट उत्तर सापडले नाही",
    "नहीं मिला",
)


def is_refusal(answer: str) -> bool:
    answer_lc = answer.lower()
    return any(marker.lower() in answer_lc for marker in REFUSAL_MARKERS)


def build_eval_store():
    docs = []
    resume = Path("data/uploads/Anuj-Chauhan-resume.pdf")
    paper = Path("data/uploads/MosquitoFusion_Improved_Research_Paper.docx")
    if resume.exists():
        docs.extend(extract_pdf_documents(resume, resume.name))
    if paper.exists():
        docs.extend(extract_docx_documents(paper, paper.name))
    if not docs:
        raise RuntimeError("Expected eval files in data/uploads.")
    return build_vector_store(build_chunks(docs))


def main() -> None:
    vector_store = build_eval_store()
    passed = 0
    hallucination_fail = 0
    refusal_pass = 0
    refusal_total = 0

    for case in CASES:
        answer, docs = answer_question(vector_store, case.question, [])
        answer_lc = answer.lower()
        refused = is_refusal(answer)
        has_required = all(term.lower() in answer_lc for term in case.required)
        avoids_forbidden = all(term.lower() not in answer_lc for term in case.forbidden)
        if case.should_refuse:
            refusal_total += 1
            ok = refused
            refusal_pass += int(ok)
        else:
            ok = has_required and avoids_forbidden and not refused
        hallucination_fail += int(not avoids_forbidden)
        passed += int(ok)
        sources = ", ".join(f"{doc.metadata.get('source')} p.{doc.metadata.get('page')}" for doc in docs[:3])
        compact = " ".join(answer.split())
        if len(compact) > 220:
            compact = compact[:217] + "..."
        print(f"{'PASS' if ok else 'FAIL'} {case.name}: {compact} | {sources}")

    total = len(CASES)
    print(f"overall_accuracy={passed / total:.3f} ({passed}/{total})")
    if refusal_total:
        print(f"refusal_accuracy={refusal_pass / refusal_total:.3f} ({refusal_pass}/{refusal_total})")
    print(f"hallucination_rate={hallucination_fail / total:.3f} ({hallucination_fail}/{total})")


if __name__ == "__main__":
    main()
