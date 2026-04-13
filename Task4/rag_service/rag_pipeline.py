import time

from embeddings import EmbeddingModel
from vectorstore import VectorStore
from llm import QwenLLM
from fewshot import get_few_shot_examples

class RAGPipeline:
    def __init__(self):
        self.embedder = EmbeddingModel()
        self.vstore = VectorStore()
        self.llm = QwenLLM()

    def build_prompt(self, query, context_chunks):
        context = "\n".join(context_chunks)
        few_shot = get_few_shot_examples()

        system_prompt = """
You're the assistant who thinks first and then answers.
Always write down your steps.

Format:
1. ...
2. ...
3. Therefore, the answer is ...
"""

        prompt = f"""
{system_prompt}

Context:
{context}

Examples:
{few_shot}

Q: {query}
A:
"""
        return prompt


    def run(self, query, max_distance: float = 0.85, top_k: int = 5):
        query_embedding = self.embedder.encode(query)
        results = self.vstore.search(query_embedding, top_k)

        docs = results.get("documents", [[]])[0]
        dists = results.get("distances", [[]])[0]

        valid_docs = [doc for doc, d in zip(docs, dists) if d <= max_distance]

        if not valid_docs:
            valid_docs = docs[:top_k]

        if not valid_docs:
            return {
                "query": query,
                "context": [],
                "answer": "Не найдено релевантных фрагментов для ответа."
            }

        prompt = self.build_prompt(query, valid_docs)
        answer = self.llm.generate(prompt)

        return {
            "query": query,
            "context": valid_docs,
            "answer": answer
        }
