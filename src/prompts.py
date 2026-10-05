ESCAPE_PHRASE = "Not covered in the provided material."

GROUNDING = (
    "Answer ONLY from the passages below. "
    f"If the answer is not there, reply exactly: {ESCAPE_PHRASE}"
)

TEMPLATE = """{grounding}

<passages>
{context}
</passages>

<question>
{question}
</question>

{grounding}"""
