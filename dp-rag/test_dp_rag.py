import os
from collections import Counter
import json
from termcolor import colored, cprint
from pup_vector_store import PUPVectorStore, PUPVectorStoreConfig
from dp_model import DPModel, DPGenerationConfig
from test_data import print_items, medical_dirichlet_documents, medical_dirichlet_full
from dp_rag_engine import DPRAGEngine, DPGenerationConfig, PUPVectorStore, PUPVectorStoreConfig

class Evaluator:
    def __init__(self):
        self.counter = Counter()
        
    def symptoms(self, disease: str, epsilon: float, success: bool):
        self.counter[json.dumps(None)] += 1
        self.counter[json.dumps(("*", "*"))] += 1
        self.counter[json.dumps(("symptoms", "*"))] += 1
        self.counter[json.dumps(("symptoms", "*", epsilon, "success"))] += 1 if success else 0
        self.counter[json.dumps(("symptoms", disease,))] += 1
        self.counter[json.dumps(("symptoms", disease, epsilon, "success"))] += 1 if success else 0
    
    def dump(self):
        os.makedirs('results', exist_ok=True)
        with open(f"results/{self.counter[json.dumps(None)]}_evaluation.json", 'w') as f:
            json.dump(self.counter, f, indent=2)
        with open(f"results/evaluation.json", 'w') as f:
            json.dump(self.counter, f, indent=2)
    
    def load(self):
        try:
            with open(f"results/evaluation.json", 'r') as f:
                self.counter = Counter(json.load(f))
        except Exception:
            print("No file to load")


class MedicalRAGTests:
    def __init__(self):
        # Use data from Huggingface
        self.docs = medical_dirichlet_documents()
        # Setup the DP RAG Engine
        self.dre = DPRAGEngine(
            pup_vector_store_config = PUPVectorStoreConfig(
                # top_k = 5,
                # epsilon=0.1,
                top_p = 0.02,
                epsilon=0.5,
                # differential_pivacy=False,
            ),
            dp_generation_config=DPGenerationConfig(
                temperature=1.0,
                max_new_tokens=70,
                alpha = 1.0,
                omega = 0.01,
                epsilon = 5.0,
                # differential_pivacy=False,
            ),
        )
        # Add all docs to RAG engine
        for doc in self.docs:
            self.dre.add(doc)

    def test_symptoms(self):
        evaluator = Evaluator()
        evaluator.load()
        for i, data in enumerate(medical_dirichlet_full()):
            if i<evaluator.counter["null"]:
                cprint(f"Skip experiment {i} on {evaluator.counter["null"]}", "red")
                continue
            question  = f"I am experiencing the following symptoms: {', '.join(data['symptom'])}. What is my disease?"
            answer = self.dre.dp_chat(question)
            disease = data['disease']
            count = evaluator.counter[json.dumps(("*", "*"))]
            cprint(count, 'yellow')
            cprint(question, 'white')
            cprint(disease, 'grey')
            success = disease in answer
            epsilon = round(self.dre.privacy_loss_distribution.get_epsilon_for_delta(0.001), 1)
            cprint(answer, 'green' if success else 'red')
            evaluator.symptoms(disease, epsilon, success)
            if (count+1) % 100 == 0:
                evaluator.dump()
            if (count+1) % 50 == 0:
                cprint(json.dumps(evaluator.counter, indent=2), 'dark_grey')
        with open(f'out/test_symptoms_{count}.json', 'w') as f:
            evaluator.dump()
        cprint(evaluator, "yellow")


if __name__ == "__main__":
    # Launch this with 'nohup python test_dp_rag.py 2>&1 | tee -a out/output.log &'
    mrt = MedicalRAGTests()
    mrt.test_symptoms()