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

        system_prompt = f"""
        You are an expert assistant that answers STRICTLY based on the provided context.
        NEVER use external knowledge, assumptions, or pre-trained facts.
        If the context does not contain enough information, respond EXACTLY with:
        "I cannot answer based on the provided context."
        Never invent facts, numbers, or details not present in the context.
        Do NOT repeat the same point. Each reasoning step must introduce NEW information.
        Limit reasoning to 3–4 concise steps. Stop when you have enough to answer.

        Here are some examples:

        {few_shot}
        
        ---
        Here is context:
        {context}

        Now solve the following:

        Question: {query}

        Answer:
        Let's think step by step.
        
        Reasoning:"""
        return system_prompt


    def run(self, query, max_distance: float = 0.65, top_k: int = 5):
        query_embedding = self.embedder.encode(query)
        results = self.vstore.search(query_embedding, top_k)

        docs = results.get("documents", [])
        dists = results.get("distances", [])

        valid_docs = [doc for doc, d in zip(docs, dists) if d <= max_distance]

        if not valid_docs:
            valid_docs = docs[:top_k]

        if not valid_docs:
            return {
                "query": query,
                "context": [],
                "answer": "No relevant fragments found to answer the question."
            }

        prompt = self.build_prompt(query, valid_docs)
        raw_answer = self.llm.generate(prompt)

        answer = self._extract_final_answer(raw_answer)

        return {
            "query": query,
            "context": valid_docs,
            "answer": answer,
            "raw_answer": raw_answer,
        }

    @staticmethod
    def _extract_final_answer(text: str) -> str:
        if "Reasoning:" in text:
            return text.split("Reasoning:")[-1].strip()
        return text.strip()