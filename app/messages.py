"""Every sentence an employee can see when we can't give a normal answer.

One place, so the wording is consistent (PS.md R9, R10) and easy to change.
Written for a tired employee at 8 PM: what happened, and what to do next.
"""

NOT_COVERED = (
    "Your company's transport policy doesn't cover this. "
    "Please contact your transport desk for help."
)

TRY_AGAIN = (
    "We couldn't get an answer right now. Please try again in a moment. "
    "If it keeps happening, contact your transport desk."
)

INTERNAL_ERROR = "Something went wrong on our side. Please try again in a moment."

QUESTION_LENGTH = "Please enter a question between {min} and {max} characters."
