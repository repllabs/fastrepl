<h1 align="center">⚡♾️ FastREPL</h1>
    <p align="center">
        <p align="center">Fast Run-Eval-Polish Loop for LLM Applications.</p>
        <p align="center">
          <strong>
            This project is still in the early development stage. Have questions? <a href="https://calendly.com/yujonglee/fastrepl">Let's chat!</a>
          </strong>
        </p>
    </p>
<h4 align="center">
    <a href="https://github.com/fastrepl/fastrepl/actions/workflows/ci.yaml" target="_blank">
        <img src="https://github.com/fastrepl/fastrepl/actions/workflows/ci.yaml/badge.svg" alt="CI Status">
    </a>
    <a href="https://pypi.org/project/fastrepl" target="_blank">
        <img src="https://img.shields.io/pypi/v/fastrepl.svg" alt="PyPI Version">
    </a>
    <a href="https://discord.gg/nMQ8ZqAegc" target="_blank">
        <img src="https://dcbadge.vercel.app/api/server/nMQ8ZqAegc?style=flat">
    </a>
<!--     <a target="_blank" href="https://colab.research.google.com/github/fastrepl/fastrepl/blob/main/docs/getting_started/quickstart.ipynb">
      <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
    </a>     -->
</h4>

## Quickstart

Let's say we have this existing system:

```python
import openai

context = """
The first step is to decide what to work on. The work you choose needs to have three qualities: it has to be something you have a natural aptitude for, that you have a deep interest in, and that offers scope to do great work.
In practice you don't have to worry much about the third criterion. Ambitious people are if anything already too conservative about it. So all you need to do is find something you have an aptitude for and great interest in.
"""

def run_qa(question: str) -> str:
    return openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": f"Answer in less than 30 words. Use the following context if needed: {context}",
            },
            {"role": "user", "content": question},
        ],
    )["choices"][0]["message"]["content"]
```

We already have a fixed context. Now, let's ask some questions. `local_runner` is used here to run it locally with threads and progress tracking. We will have `remote_runner` to run the same in the cloud.

```python
contexts = [[context]] * len(questions)

# https://huggingface.co/datasets/repllabs/questions_how_to_do_great_work
questions = [
    "how to do great work?.",
    "How can curiosity be nurtured and utilized to drive great work?",
    "How does the author suggest finding something to work on?",
    "How did Van Dyck's painting differ from Daniel Mytens' version and what message did it convey?",
]

runner = fastrepl.local_runner(fn=run_qa)
ds = runner.run(args_list=[(q,) for q in questions], output_feature="answer")

ds = ds.add_column("question", questions)
ds = ds.add_column("contexts", contexts)
# fastrepl.Dataset({
#     features: ['answer', 'question', 'contexts'],
#     num_rows: 4
# })
```

Now, let's use one of our evaluators to evaluate the dataset. Note that we are running it 5 times to ensure we obtain consistent results.

```python
evaluator = fastrepl.RAGEvaluator(node=fastrepl.RAGAS(metric="Faithfulness"))

ds = fastrepl.local_runner(evaluator=evaluator, dataset=ds).run(num=5)
# ds["result"]
# [[0.25, 0.0, 0.25, 0.25, 0.5],
#  [0.5, 0.5, 0.5, 0.75, 0.875],
#  [0.66, 0.66, 0.66, 0.66, 0.66],
#  [1.0, 1.0, 1.0, 1.0, 1.0]]
```
Seems like we are getting quite good results. If we increase the number of samples a bit, we can obtain a reliable evaluation of the entire system. **We will keep working on bringing better evaluations.**

Detailed documentation is [here](https://docs.fastrepl.com/getting_started/quickstart).

## Contributing
Any kind of contribution is welcome. 

- Development: Please read [CONTRIBUTING.md](CONTRIBUTING.md) and [tests](tests).
- Bug reports: Use [Github Issues](https://github.com/yujonglee/fastrepl/issues).
- Feature request and questions: Use [Github Discussions](https://github.com/yujonglee/fastrepl/discussions).
