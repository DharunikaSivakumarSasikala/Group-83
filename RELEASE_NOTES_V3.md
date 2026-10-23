# SkillScope AI v3 Release Notes

## Completed faculty-requested changes

- Signed candidate invitation links assign the participant and role before the link is opened.
- Candidates no longer choose their technical domain or question set.
- Every ten-question set contains seven written questions and three coding/technical quizzes.
- Every question has a Skip button; skipped questions receive zero accuracy.
- Written answers use technical keyword and word-variant matching rather than grammar quality.
- Eighty percent per question is the competency threshold.
- Overall accuracy is the mean of all ten question accuracies.
- The assessment requests full screen from the candidate's Begin click and warns if it is exited.
- Submission exits full screen and opens a detailed result report in a new tab when popups are allowed.
- Results include employee answer, expected evidence or quiz explanation, missing concepts and approved study links.
- RAG generates developmental feedback for the weakest written answers while deterministic scoring remains the official percentage.
- Hallucination documentation and a human-labelled metric calculator were added.

## Verification

- 17 automated tests pass.
- An end-to-end HTTP test passed for signed-link start, ten assigned questions, three quizzes, submission, result creation and result retrieval.

## Local demo

1. `python -m pip install -r requirements.txt`
2. `ollama pull llama3.2:3b`
3. `python web_app.py`
4. In another terminal: `python create_invitation.py --participant EMP-001 --role qa_test_engineer`
5. Open the generated URL.

For a quick UI demo without waiting for Ollama feedback, start the server in PowerShell with `$env:SKILLSCOPE_RAG_MAX_QUESTIONS="0"` before running `python web_app.py`. Set it back to `2` or remove the variable to demonstrate bounded RAG feedback for the two weakest written answers.
