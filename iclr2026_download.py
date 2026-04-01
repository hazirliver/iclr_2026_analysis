import json
import time
from typing import Any, Iterable

import requests
from tqdm import tqdm

BASE_URL = "https://api.openreview.net"
LOGIN_URL = f"{BASE_URL}/login"
NOTES_URL = f"{BASE_URL}/notes"

INVITATION = "ICLR.cc/2026/Conference/-/Submission"
OUTPUT_PATH = "iclr2026_submissions_with_reviews.jsonl"

OPENREVIEW_EMAIL = "your_email_here"
OPENREVIEW_PASSWORD = "your_password_here"

REQUEST_TIMEOUT = 30
SLEEP_BETWEEN_REQUESTS_SEC = 0.2
PAGE_SIZE = 1000


def extract_field_value(field: Any) -> Any:
    if isinstance(field, dict) and "value" in field:
        return field["value"]
    return field


def build_session(email: str, password: str) -> requests.Session:
    session = requests.Session()

    response = session.post(
        LOGIN_URL,
        json={"id": email, "password": password},
        timeout=REQUEST_TIMEOUT,
    )

    if response.status_code != 200:
        raise RuntimeError(
            f"OpenReview login failed: {response.status_code} {response.text}"
        )

    return session


def is_likely_review(reply: dict[str, Any]) -> bool:
    invitation = reply.get("invitation", "") or ""
    content = reply.get("content", {}) or {}

    if invitation.endswith("/Review") or invitation.endswith("/Official_Review"):
        return True

    review_like_fields = {
        "rating",
        "confidence",
        "summary",
        "strengths",
        "weaknesses",
        "questions",
        "limitations",
        "soundness",
        "presentation",
        "contribution",
    }
    return any(key in content for key in review_like_fields)


def normalize_review(reply: dict[str, Any]) -> dict[str, Any]:
    content = reply.get("content", {}) or {}
    return {
        "id": reply.get("id"),
        "forum": reply.get("forum"),
        "replyto": reply.get("replyto"),
        "invitation": reply.get("invitation"),
        "signatures": reply.get("signatures"),
        "cdate": reply.get("cdate"),
        "tcdate": reply.get("tcdate"),
        "tmdate": reply.get("tmdate"),
        "readers": reply.get("readers"),
        "writers": reply.get("writers"),
        "content": {k: extract_field_value(v) for k, v in content.items()},
    }


def normalize_submission(note: dict[str, Any]) -> dict[str, Any]:
    content = note.get("content", {}) or {}
    details = note.get("details", {}) or {}
    direct_replies = details.get("directReplies", []) or []

    reviews = [
        normalize_review(reply) for reply in direct_replies if is_likely_review(reply)
    ]

    return {
        "id": note.get("id"),
        "forum": note.get("forum"),
        "number": note.get("number"),
        "invitation": note.get("invitation"),
        "cdate": note.get("cdate"),
        "tcdate": note.get("tcdate"),
        "tmdate": note.get("tmdate"),
        "title": extract_field_value(content.get("title")),
        "abstract": extract_field_value(content.get("abstract")),
        "authors": extract_field_value(content.get("authors")),
        "authorids": extract_field_value(content.get("authorids")),
        "keywords": extract_field_value(content.get("keywords")),
        "primary_area": extract_field_value(content.get("primary_area")),
        "venue": extract_field_value(content.get("venue")),
        "venueid": extract_field_value(content.get("venueid")),
        "pdf": extract_field_value(content.get("pdf")),
        "content": {k: extract_field_value(v) for k, v in content.items()},
        "reviews": reviews,
        "review_count": len(reviews),
    }


def request_json(
    session: requests.Session,
    url: str,
    params: dict[str, Any],
) -> dict[str, Any]:
    response = session.get(url, params=params, timeout=REQUEST_TIMEOUT)

    if response.status_code == 403:
        raise PermissionError(
            "403 Forbidden from OpenReview. "
            "Your account likely does not have access to ICLR 2026 submissions/reviews, "
            "or the venue is not public."
        )

    response.raise_for_status()
    return response.json()


def get_total_count(session: requests.Session) -> int | None:
    params = {
        "invitation": INVITATION,
        "limit": 1,
        "offset": 0,
    }
    data = request_json(session, NOTES_URL, params)
    return data.get("count")


def fetch_submission_pages(
    session: requests.Session,
) -> Iterable[list[dict[str, Any]]]:
    offset = 0

    while True:
        params = {
            "invitation": INVITATION,
            "details": "directReplies",
            "limit": PAGE_SIZE,
            "offset": offset,
        }

        data = request_json(session, NOTES_URL, params)
        notes = data.get("notes", []) or []

        if not notes:
            break

        yield notes
        offset += PAGE_SIZE
        time.sleep(SLEEP_BETWEEN_REQUESTS_SEC)


def save_iclr_2026_submissions_with_reviews(
    email: str,
    password: str,
    output_path: str = OUTPUT_PATH,
) -> None:
    session = build_session(email, password)
    total = get_total_count(session)

    total_reviews = 0
    total_submissions = 0

    with (
        open(output_path, "w", encoding="utf-8") as f,
        tqdm(
            total=total,
            desc="Fetching ICLR 2026 submissions",
            unit="submission",
        ) as pbar,
    ):
        for notes in fetch_submission_pages(session):
            for note in notes:
                record = normalize_submission(note)
                total_reviews += record["review_count"]
                total_submissions += 1

                f.write(json.dumps(record, ensure_ascii=False) + "\n")

            pbar.update(len(notes))
            pbar.set_postfix(
                submissions=total_submissions,
                reviews=total_reviews,
            )

    print(f"Saved {total_submissions} submissions to {output_path}")
    print(f"Extracted {total_reviews} reviews")


if __name__ == "__main__":
    save_iclr_2026_submissions_with_reviews(
        email=OPENREVIEW_EMAIL,
        password=OPENREVIEW_PASSWORD,
    )
