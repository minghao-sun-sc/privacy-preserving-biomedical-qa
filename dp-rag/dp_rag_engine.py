from termcolor import colored, cprint
from pup_vector_store import PUPVectorStore, PUPVectorStoreConfig
from dp_model import DPModel, DPGenerationConfig
from test_data import print_items, medical_dirichlet_documents

class DPRAGEngine:
    def __init__(
            self,
            pup_vector_store_config: PUPVectorStoreConfig = PUPVectorStoreConfig(),
            model_id="microsoft/Phi-3.5-mini-instruct",
            dp_generation_config: DPGenerationConfig = DPGenerationConfig(),
            
        ):
        self.pup_vector_store: PUPVectorStore = PUPVectorStore(config=pup_vector_store_config)
        self.dp_model: DPModel = DPModel(model_id=model_id)
        self.dp_generation_config = dp_generation_config
        self.privacy_loss_distribution = self.pup_vector_store.privacy_loss_distribution.compose(dp_generation_config.privacy_loss_distribution)
    
    def add(self, entry: str):
        self.pup_vector_store.add(entry)
    
    def pup_retrieve(self, query: str) -> list[str]:
        return self.pup_vector_store.pup_retrieve(query=query)

    def dp_chat(self, question: str) -> str:
        retrieved_documents = self.pup_vector_store.pup_retrieve(question)
        # DEBUG
        # print(len(retrieved_documents))
        # print_items(retrieved_documents[:3], ['dark_grey', 'light_grey'])
        return self.dp_model.dp_chat(
            retrieved_documents,
            question,
            self.dp_generation_config,
        )


def main():
    # Use data from Huggingface
    docs = medical_dirichlet_documents()
    # Setup the DP RAG Engine
    dre = DPRAGEngine(
        pup_vector_store_config = PUPVectorStoreConfig(
            # top_k = 5,
            # epsilon=0.1,
            top_p = 0.02,
            epsilon=0.2,
            # differential_pivacy=False,
        ),
        dp_generation_config=DPGenerationConfig(
            temperature=1.0,
            max_new_tokens=70,
            alpha = 1.0,
            omega = 0.01,
            epsilon = 3.0,
            # differential_pivacy=False,
        ),
    )
    # Add all docs to RAG engine
    for doc in docs:
        dre.add(doc)

    for i, question in enumerate([
        "What is Anna Svenson diagnosed with?",# Snurflaxitis
        "What is Jesper Mortensen diagnosed with?",# Sudden Onset Disco Toe Displacement Syndrome SODTLS
        "What is Olaf Larsen diagnosed with?",# Glibblewarts
        "What is Erich Meyer diagnosed with?",# Burpdynia
        "Who has Snurflaxitis?",# Snurflaxitis
        "Who has Sudden Onset Disco Toe Displacement Syndrome?",# Sudden Onset Disco Toe Displacement Syndrome SODTLS
        "Who has Glibblewarts?",# Glibblewarts
        "Who has Burpdynia?",# Burpdynia
        "Given the following symptoms: Feverish cough, Sore throat, Swollen lymph nodes, Muscle weakness, what is the disease?",
        "What are the symptoms associated with Snurflaxitis?",
        "When should the treatment: Flarglepox Discombobulation be used? Which disease?",
        "I'm experiencing Uncontrollable taco cravings, severe digestive contortions and unrelenting cravings for salsa. What should I do?",
        "I'm experiencing weird symptoms: Toe numbness, Painful toe curling, Limited toe extension, and Glittering nail growth. What's wrong with me? What disgnosis?",
    ]):
        print(f"Question n°{i+1}")
        answer = dre.dp_chat(question)
        cprint(question, 'red')
        cprint(answer, 'blue')
        cprint(dre.privacy_loss_distribution.get_epsilon_for_delta(1e-3), 'yellow')

if __name__ == "__main__":
    main()