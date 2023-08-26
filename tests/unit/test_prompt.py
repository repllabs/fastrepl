import outlines.text as text


class TestPromptTry:
    def test_basic(self):
        @text.prompt
        def prompt(instruction, examples, question):
            """{{ instruction }}

            {% for example in examples %}
            Q: {{ example.question }}
            A: {{ example.answer }}
            {% endfor %}
            Q: {{ question }}
            """

        assert (
            prompt("Instruction", [], "Question")
            == """Instruction

Q: Question"""
        )

        assert (
            prompt(instruction="Instruction", examples=[], question="Question")
            == """Instruction

Q: Question"""
        )

        assert (
            prompt(
                instruction="Instruction",
                examples=[
                    {"question": "q1", "answer": "a1"},
                    {"question": "q2", "answer": "a2"},
                ],
                question="a3",
            )
            == """Instruction

Q: q1
A: a1
Q: q2
A: a2
Q: a3"""
        )

    def test_if(self):
        @text.prompt
        def prompt(question, context=""):
            """{% if context != '' %}
            Consider the following context when answering the question:
            {{ context }}\n
            {% endif %}
            Q: {{ question }}
            """

        assert prompt("hello?", "") == """Q: hello?"""
        assert prompt("hello?") == """Q: hello?"""
        assert prompt(question="hello?") == """Q: hello?"""

        assert (
            prompt("hello?", context="this is context")
            == """Consider the following context when answering the question:
this is context

Q: hello?"""
        )
