
def build_prompt_text(example, kind):
    if kind == "ag_news":
        return (
            f"Classify the news headline into one of the categories.\n"
            f"Headline: {example['text']}\nLabel:"
        )

    if kind == "arc_easy":
        q = example["question"]
        choices = "\n".join([f"({c['label']}) {c['text']}" for c in example["choices"]])
        return f"Answer the multiple-choice science question.\nQ: {q}\n{choices}\nA:"

    if kind == "hellaswag":
        ctx = example["ctx"]
        choices = example["endings"]
        chs = "\n".join([f"({i}) {c}" for i, c in enumerate(choices)])
        return f"Choose the most plausible continuation.\nContext: {ctx}\n{chs}\nAnswer index:"

    if kind == "gsm8k":
        return (
            "Solve step by step and give the final answer as 'Answer: <number>'.\n"
            f"Problem: {example['question']}"
        )

    return str(example)

def build_prompt_vqa(question):
    return f"Answer the question about the image concisely.\nQ: {question}\nA:"


def build_prompt_caption():
    return "Describe the image in one or two sentences."
