"""Every sentence an employee can see when we can't give a normal answer.

One place, so the wording is consistent and easy to change: what happened, and
what to do next (usually: contact the transport desk).
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

# A guardrail stopped the request (an attempt to change the assistant's rules,
# or a request for another company's policy).
GUARDRAIL = ("I can only answer questions about your own company's transport policy. "
             "For anything else, please contact your transport desk.")
