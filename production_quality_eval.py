from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from app import (
    answer_question,
    build_chunks,
    build_vector_store,
    Document,
    extract_docx_documents,
    extract_pdf_documents,
    extract_fee_records,
    infer_document_metadata,
    save_document_metadata,
    save_fee_records,
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


def build_followup_store():
    doc = Document(
        page_content=(
            "Cold Mailing Guide\n"
            "Week 1: Research your target audience, build an ICP, and collect verified leads.\n"
            "Week 2: Write short personalized cold email copy and prepare subject line variants.\n"
            "Week 3: Launch the first campaign, send a small batch of personalized emails, "
            "track open and reply rates, follow up with non-responders, and improve the message based on replies.\n"
            "Week 4: Scale the campaign, test new segments, and document the winning templates."
        ),
        metadata={"source": "cold_mailing_guide.txt", "page": 1, "file_type": "txt"},
    )
    return build_vector_store(build_chunks([doc]))


def build_fee_store():
    doc = Document(
        page_content=(
            "M.Arch. 1st Year session 2026-27\n"
            "Fee Structure\n"
            "M.Arch fee: Rs. 73,200 with caution money Rs. 5,000.\n"
            "Hostel fee: Rs. 27,000 with security money Rs. 2,000.\n"
            "This fee notice is for M.Arch students only."
        ),
        metadata={"source": "MArch_fee_notice.pdf", "page": 1, "file_type": "pdf"},
    )
    docs = [doc]
    metadata = infer_document_metadata(docs)
    records = extract_fee_records(docs, metadata)
    save_document_metadata(metadata)
    save_fee_records(records)
    return build_vector_store(build_chunks(docs))


def build_btech_tuition_only_store():
    doc = Document(
        page_content=(
            "B.Tech Admission Fee Notice 2026-27\n"
            "Fee Structure\n"
            "B.Tech tuition fee: \u20b988,000 per year.\n"
            "No hostel fee is listed in this notice."
        ),
        metadata={"source": "BTech_fee_notice.pdf", "page": 1, "file_type": "pdf"},
    )
    docs = [doc]
    metadata = infer_document_metadata(docs)
    records = extract_fee_records(docs, metadata)
    save_document_metadata(metadata)
    save_fee_records(records)
    return build_vector_store(build_chunks(docs))


def build_multi_program_fee_store():
    docs = [
        Document(
            page_content=(
                "B.Tech Fee Structure 2026-27\n"
                "B.Tech course fee: Rs. 88,000.\n"
                "B.Tech hostel fee: Rs. 12,000."
            ),
            metadata={"source": "BTech_fee_notice.pdf", "page": 1, "file_type": "pdf"},
        ),
        Document(
            page_content=(
                "MBA Fee Structure 2026-27\n"
                "MBA course fee: Rs. 95,000.\n"
                "MBA hostel fee: Rs. 18,000."
            ),
            metadata={"source": "MBA_fee_notice.pdf", "page": 1, "file_type": "pdf"},
        ),
    ]
    metadata = infer_document_metadata(docs)
    records = extract_fee_records(docs, metadata)
    save_document_metadata(metadata)
    save_fee_records(records)
    return build_vector_store(build_chunks(docs))


def build_marathi_notice_store():
    doc = Document(
        page_content=(
            "एम.प्लॅन प्रथम वर्ष निवड यादी सूचना\n"
            "ही सूचना एम.प्लॅन प्रथम वर्ष प्रवेशासाठी निवड झालेल्या विद्यार्थ्यांसाठी जारी करण्यात आली आहे.\n"
            "निवड झालेल्या विद्यार्थ्यांचा निकाल PBEL 3.0 पोर्टलवर उपलब्ध करून देण्यात आला आहे."
        ),
        metadata={"source": "selected_mplan.pdf", "page": 1, "file_type": "pdf"},
    )
    return build_vector_store(build_chunks([doc]))


def build_marathi_fee_store():
    doc = Document(
        page_content=(
            "एम.आर्क प्रथम वर्ष सत्र 2026-27 फी नोटीस\n"
            "ही फी नोटीस एम.आर्क अभ्यासक्रमासाठी आहे.\n"
            "कॉलेज व विद्यापीठ शुल्क: \u20b973,200\n"
            "वसतिगृह शुल्क: \u20b927,000"
        ),
        metadata={"source": "march_fee_26_27.pdf", "page": 1, "file_type": "pdf"},
    )
    docs = [doc]
    metadata = infer_document_metadata(docs)
    records = extract_fee_records(docs, metadata)
    save_document_metadata(metadata)
    save_fee_records(records)
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

    followup_store = build_followup_store()
    first_answer, first_docs = answer_question(followup_store, "What are the steps for cold mailing? Guide me week by week.", [])
    history = [
        {"role": "user", "content": "What are the steps for cold mailing? Guide me week by week."},
        {
            "role": "assistant",
            "content": first_answer,
            "sources": [f"{doc.metadata.get('source')} p.{doc.metadata.get('page')}" for doc in first_docs],
        },
    ]
    followup_answer, _ = answer_question(followup_store, "explain me the week 3", history)
    followup_ok = all(term in followup_answer.lower() for term in ("week 3", "campaign", "follow"))
    print(f"{'PASS' if followup_ok else 'FAIL'} followup_week_3: {' '.join(followup_answer.split())}")

    fee_store = build_fee_store()
    btech_answer, _ = answer_question(fee_store, "tell me the fee structure of btech and its hostel fee", [])
    btech_ok = "not b.tech" in btech_answer.lower() or "cannot answer b.tech" in btech_answer.lower()
    print(f"{'PASS' if btech_ok else 'FAIL'} fee_mismatch_btech_from_march: {' '.join(btech_answer.split())}")

    march_answer, _ = answer_question(fee_store, "tell me the fee structure of M.Arch and its hostel fee", [])
    march_ok = all(term.lower() in march_answer.lower() for term in ("m.arch", "73,200", "27,000"))
    print(f"{'PASS' if march_ok else 'FAIL'} fee_positive_march: {' '.join(march_answer.split())}")

    btech_tuition_store = build_btech_tuition_only_store()
    rupee_answer, _ = answer_question(btech_tuition_store, "what is the B.Tech tuition fee", [])
    rupee_ok = "b.tech" in rupee_answer.lower() and "88,000" in rupee_answer
    print(f"{'PASS' if rupee_ok else 'FAIL'} fee_rupee_symbol_btech: {' '.join(rupee_answer.split())}")

    missing_hostel_answer, _ = answer_question(btech_tuition_store, "what is the B.Tech hostel fee", [])
    missing_hostel_ok = "not a clearly supported hostel fee" in missing_hostel_answer.lower()
    print(f"{'PASS' if missing_hostel_ok else 'FAIL'} fee_missing_type_btech: {' '.join(missing_hostel_answer.split())}")

    multi_fee_store = build_multi_program_fee_store()
    ambiguous_answer, _ = answer_question(multi_fee_store, "what is the hostel fee", [])
    ambiguous_ok = "multiple programs" in ambiguous_answer.lower() and "specify the program" in ambiguous_answer.lower()
    print(f"{'PASS' if ambiguous_ok else 'FAIL'} fee_ambiguous_multi_program: {' '.join(ambiguous_answer.split())}")

    marathi_notice_store = build_marathi_notice_store()
    notice_answer, _ = answer_question(
        marathi_notice_store,
        "ही सूचना कोणत्या विद्यार्थ्यांसाठी जारी करण्यात आली आहे? निकाल कोणत्या पोर्टलवर उपलब्ध करून देण्यात आला आहे?",
        [],
    )
    notice_ok = "एम.प्लॅन" in notice_answer and "PBEL 3.0" in notice_answer and "पोर्टल" in notice_answer
    print(f"{'PASS' if notice_ok else 'FAIL'} marathi_notice_students_portal: {' '.join(notice_answer.split())}")

    marathi_fee_store = build_marathi_fee_store()
    marathi_fee_answer, _ = answer_question(
        marathi_fee_store,
        "ही फी नोटीस कोणत्या अभ्यासक्रमासाठी आहे? कॉलेज व विद्यापीठ शुल्क किती आहे?",
        [],
    )
    marathi_fee_ok = "M.Arch" in marathi_fee_answer and "73,200" in marathi_fee_answer
    print(f"{'PASS' if marathi_fee_ok else 'FAIL'} marathi_fee_course_college_university: {' '.join(marathi_fee_answer.split())}")


if __name__ == "__main__":
    main()
