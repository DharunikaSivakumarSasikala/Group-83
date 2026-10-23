"""Create a signed SkillScope candidate assessment link."""

from __future__ import annotations

import argparse
import time
from urllib.parse import quote

from skillscope_core import create_invitation_token


ROLES = (
    "backend_software_engineer",
    "frontend_ui_developer",
    "devops_sre_engineer",
    "qa_test_engineer",
    "data_engineer",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--participant", required=True, help="Anonymous employee ID")
    parser.add_argument("--role", required=True, choices=ROLES)
    parser.add_argument("--days", type=int, default=7, help="Link validity in days")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    args = parser.parse_args()
    expires = int(time.time()) + max(1, args.days) * 86400
    token = create_invitation_token(args.participant, args.role, expires)
    print(f"{args.base_url.rstrip('/')}/assessment?token={quote(token)}")


if __name__ == "__main__":
    main()
