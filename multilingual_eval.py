from __future__ import annotations

from dataclasses import dataclass

from app import answer_question, load_vector_store


@dataclass(frozen=True)
class AnswerCase:
    name: str
    question: str
    required_terms: tuple[str, ...] = ()
    forbidden_terms: tuple[str, ...] = ()
    should_refuse: bool = False


CASES = [
    AnswerCase(
        name="mr_giver",
        question="हा शुभसंदेश कोणी दिला आहे?",
        required_terms=("विनोद तावडे",),
    ),
    AnswerCase(
        name="mr_role",
        question="विनोद तावडे कोणत्या पदावर कार्यरत होते?",
        required_terms=("मंत्री", "शालेय शिक्षण", "वैद्यकीय शिक्षण", "मराठी भाषा"),
    ),
    AnswerCase(
        name="mr_date",
        question="शुभसंदेशाची तारीख काय आहे?",
        required_terms=("१७ मार्च", "२०१५"),
        forbidden_terms=("२०१८", "२७ मार्च"),
    ),
    AnswerCase(
        name="mr_institutions",
        question="हा शुभसंदेश कोणत्या संस्थेसाठी आहे?",
        required_terms=("वैद्य साने ट्रस्ट", "माधवबाग"),
    ),
    AnswerCase(
        name="mr_field",
        question="शुभसंदेशात कोणत्या क्षेत्रातील कार्याचे कौतुक करण्यात आले आहे?",
        required_terms=("आरोग्य क्षेत्रातील कार्य",),
    ),
    AnswerCase(
        name="mr_campaign",
        question="शुभसंदेशात कोणत्या अभियानाचा उल्लेख आहे?",
        required_terms=("आरोग्यं हृदयसंपदा",),
        forbidden_terms=("हृदयसंपन्न",),
    ),
    AnswerCase(
        name="mr_diseases",
        question="या शुभसंदेशानुसार कोणत्या दोन आजारांचे प्रमाण वाढत आहे?",
        required_terms=("हृदयरोग", "मधुमेह"),
    ),
    AnswerCase(
        name="mr_unsupported_age",
        question="विनोद तावडे यांचे वय किती आहे?",
        should_refuse=True,
    ),
    AnswerCase(
        name="hi_giver",
        question="यह शुभसंदेश किसने दिया है?",
        required_terms=("विनोद तावडे",),
    ),
    AnswerCase(
        name="hi_date",
        question="इस शुभसंदेश की तारीख क्या है?",
        required_terms=("१७ मार्च", "२०१५"),
        forbidden_terms=("२०१८", "२७ मार्च"),
    ),
    AnswerCase(
        name="hi_unsupported_place",
        question="विनोद तावडे कहाँ पैदा हुए थे?",
        should_refuse=True,
    ),
    AnswerCase(
        name="en_giver",
        question="Who gave this message?",
        required_terms=("Vinod Tawde",),
    ),
    AnswerCase(
        name="en_date",
        question="What is the date of this message?",
        required_terms=("१७ मार्च", "२०१५"),
        forbidden_terms=("2018", "27 March"),
    ),
    AnswerCase(
        name="en_unsupported_salary",
        question="What was Vinod Tawde's salary?",
        should_refuse=True,
    ),
    AnswerCase(
        name="docx_abstract_typo",
        question="what is the abstract of mosquitp",
        required_terms=("MosquitoFusion", "1,204", "YOLOv8s", "GIS"),
        forbidden_terms=("LIC HFL", "Junior Assistant"),
    ),
    AnswerCase(
        name="docx_methodology_typo",
        question="give me the methodology og that resarch paper",
        required_terms=("Overall Pipeline", "Data Preprocessing", "YOLOv8s", "GIS Integration"),
        forbidden_terms=("LIC HFL", "Junior Assistant"),
    ),
]


REFUSAL_MARKERS = (
    "could not find",
    "not found",
    "not clearly supported",
    "स्पष्ट उत्तर सापडले नाही",
    "नहीं मिला",
    "नहीं मिल",
)


def is_refusal(answer: str) -> bool:
    answer_lc = answer.lower()
    return any(marker in answer_lc for marker in REFUSAL_MARKERS)


def main() -> None:
    vector_store = load_vector_store()
    if vector_store is None:
        raise RuntimeError("No FAISS index found. Build the knowledge base first.")

    answer_pass = 0
    refusal_pass = 0
    hallucination_fail = 0

    for case in CASES:
        answer, _docs = answer_question(vector_store, case.question, [])
        answer_lc = answer.lower()
        has_required = all(term.lower() in answer_lc for term in case.required_terms)
        avoids_forbidden = all(term.lower() not in answer_lc for term in case.forbidden_terms)
        refused = is_refusal(answer)

        if case.should_refuse:
            ok = refused
            refusal_pass += int(ok)
            hallucination_fail += int(not ok)
        else:
            ok = has_required and avoids_forbidden and not refused
            answer_pass += int(ok)
            hallucination_fail += int(not avoids_forbidden)

        compact = " ".join(answer.split())
        if len(compact) > 220:
            compact = compact[:217] + "..."
        print(f"{'PASS' if ok else 'FAIL'} {case.name}: {compact}")

    total = len(CASES)
    supported_total = sum(1 for case in CASES if not case.should_refuse)
    refusal_total = total - supported_total
    print(f"supported_answer_accuracy={answer_pass / supported_total:.3f} ({answer_pass}/{supported_total})")
    print(f"unsupported_refusal_accuracy={refusal_pass / refusal_total:.3f} ({refusal_pass}/{refusal_total})")
    print(f"hallucination_rate={hallucination_fail / total:.3f} ({hallucination_fail}/{total})")


if __name__ == "__main__":
    main()
