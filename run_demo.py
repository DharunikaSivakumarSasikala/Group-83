"""One-command local demo launcher for SkillScope AI.

Usage:
    python run_demo.py

Optional:
    python run_demo.py --participant DEMO-002 --role data_engineer

The launcher:
1. Uses embedding retrieval by default.
2. Starts web_app.py if port 8000 is not already serving.
3. Creates a signed role-specific invitation with create_invitation.py.
4. Opens the generated assessment link in the default browser.
5. Keeps the web server running until Ctrl+C when this script started it.

This is only a developer/demo convenience. It does not bypass or replace
SkillScope's signed invitation flow.
"""

from __future__ import annotations

import argparse
import os
import re
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
import webbrowser
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent
WEB_APP = PROJECT_DIR / "web_app.py"
INVITATION_SCRIPT = PROJECT_DIR / "create_invitation.py"
HOST = "127.0.0.1"
PORT = 8000
BASE_URL = f"http://{HOST}:{PORT}"

ROLES = (
    "backend_software_engineer",
    "frontend_ui_developer",
    "devops_sre_engineer",
    "qa_test_engineer",
    "data_engineer",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Start SkillScope AI and open a signed demo assessment."
    )
    parser.add_argument(
        "--participant",
        default="DEMO-001",
        help="Anonymous demo participant ID (default: DEMO-001).",
    )
    parser.add_argument(
        "--role",
        choices=ROLES,
        default="qa_test_engineer",
        help="Technical role for the demo assessment.",
    )
    parser.add_argument(
        "--retriever",
        choices=("embedding", "tfidf"),
        default="embedding",
        help="Retriever to use (default: embedding).",
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Print the invitation link without opening a browser.",
    )
    return parser.parse_args()


def ensure_project_files() -> None:
    missing = [
        str(path.name)
        for path in (WEB_APP, INVITATION_SCRIPT)
        if not path.exists()
    ]
    if missing:
        raise SystemExit(
            "Missing required project file(s): " + ", ".join(missing) + "\n"
            "Place run_demo.py inside the SkillScope_AI_Prototype folder."
        )


def port_is_open() -> bool:
    try:
        with socket.create_connection((HOST, PORT), timeout=0.5):
            return True
    except OSError:
        return False


def skill_scope_is_serving() -> bool:
    try:
        with urllib.request.urlopen(BASE_URL, timeout=1.5) as response:
            body = response.read(4096).decode("utf-8", errors="ignore")
            return response.status == 200 and "SkillScope AI" in body
    except (urllib.error.URLError, TimeoutError, OSError):
        return False


def wait_for_server(process: subprocess.Popen[bytes], timeout: float = 60.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if process.poll() is not None:
            raise SystemExit(
                f"SkillScope web server stopped unexpectedly with exit code "
                f"{process.returncode}."
            )
        if skill_scope_is_serving():
            return
        time.sleep(0.25)
    raise SystemExit(
        f"Timed out waiting for SkillScope at {BASE_URL}."
    )


def start_server(retriever: str) -> tuple[subprocess.Popen[bytes] | None, bool]:
    if port_is_open():
        if skill_scope_is_serving():
            print(f"SkillScope is already running at {BASE_URL}")
            return None, False
        raise SystemExit(
            f"Port {PORT} is already in use by another application.\n"
            f"Close that application, then run this command again."
        )

    env = os.environ.copy()
    env["SKILLSCOPE_RETRIEVER"] = retriever

    print(f"Starting SkillScope with retriever: {retriever}")
    process = subprocess.Popen(
        [sys.executable, str(WEB_APP)],
        cwd=str(PROJECT_DIR),
        env=env,
    )
    wait_for_server(process)
    print(f"SkillScope is ready at {BASE_URL}")
    return process, True


def create_invitation(participant: str, role: str) -> str:
    completed = subprocess.run(
        [
            sys.executable,
            str(INVITATION_SCRIPT),
            "--participant",
            participant,
            "--role",
            role,
        ],
        cwd=str(PROJECT_DIR),
        capture_output=True,
        text=True,
    )

    output = "\n".join(
        part.strip()
        for part in (completed.stdout, completed.stderr)
        if part and part.strip()
    )

    if completed.returncode != 0:
        raise SystemExit(
            "Could not create the SkillScope invitation.\n\n" + output
        )

    # create_invitation.py prints a signed localhost assessment URL.
    matches = re.findall(r"https?://[^\s]+", output)
    if not matches:
        raise SystemExit(
            "The invitation script ran, but no assessment URL was found in "
            "its output.\n\nOutput:\n" + output
        )

    return matches[-1].rstrip(".,);]\"'")


def stop_server(process: subprocess.Popen[bytes] | None) -> None:
    if process is None or process.poll() is not None:
        return

    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


def main() -> int:
    args = parse_args()
    ensure_project_files()

    server_process: subprocess.Popen[bytes] | None = None
    started_here = False

    try:
        server_process, started_here = start_server(args.retriever)
        invitation_url = create_invitation(args.participant, args.role)

        print("\n" + "=" * 72)
        print("SKILLSCOPE DEMO READY")
        print("=" * 72)
        print(f"Participant : {args.participant}")
        print(f"Role        : {args.role}")
        print(f"Retriever   : {args.retriever}")
        print(f"Assessment  : {invitation_url}")

        if not args.no_browser:
            opened = webbrowser.open(invitation_url, new=2)
            print(
                "Browser     : opened automatically"
                if opened
                else "Browser     : could not open automatically; use the URL above"
            )

        if started_here:
            print("\nSkillScope is running. Press Ctrl+C when you are finished.")
            while server_process is not None and server_process.poll() is None:
                time.sleep(0.5)
        else:
            print("\nThe existing SkillScope server was left running.")

        return 0

    except KeyboardInterrupt:
        print("\nStopping SkillScope demo...")
        return 0
    finally:
        if started_here:
            stop_server(server_process)


if __name__ == "__main__":
    raise SystemExit(main())
