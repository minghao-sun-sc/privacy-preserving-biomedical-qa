import numpy as np
from datasets import load_dataset
from faker import Faker
fake = Faker()
from termcolor import colored, cprint

def print_items(items: list[str], colors: list[str] = ('white', 'grey')):
    for i, item in enumerate(items):
        cprint(f'- {item}\n', colors[i%2])

def simple_medical_messages():
    return [
        [
            {"role": "system", "content": """You just give to the point responses and base them on this <text> only:
            <text>The patient has symptoms. It is is diagnosed a Disease.<text>"""},
            {"role": "user", "content": """In the <text> when the patient has red eyes. What is he diagnosed with?"""},
        ],[
            {"role": "system", "content": """You just give to the point responses and base them on this <text> only:
            <text>The patient is Bob. He has empty eyes. He was diagnosed with Blocratis.<text>"""},
            {"role": "user", "content": """In the <text> when the patient has red eyes. What is he diagnosed with?"""},
        ],[
            {"role": "system", "content": """You just give to the point responses and base them on this <text> only:
            <text>The patient is Charles. He has red eyes. He was diagnosed with Avlablatosis.<text>"""},
            {"role": "user", "content": """In the <text> when the patient has red eyes. What is he diagnosed with?"""},
        ],[
            {"role": "system", "content": """You just give to the point responses and base them on this <text> only:
            <text>The patient is Donald. He has red eyes. He was diagnosed with Avlablatosis.<text>"""},
            {"role": "user", "content": """In the <text> when the patient has red eyes. What is he diagnosed with?"""},
        ],[
            {"role": "system", "content": """You just give to the point responses and base them on this <text> only:
            <text>The patient is Eve. She has red eyes. She was diagnosed with Avlablatosis.<text>"""},
            {"role": "user", "content": """In the <text> when the patient has red eyes. What is he diagnosed with?"""},
        ],[
            {"role": "system", "content": """You just give to the point responses and base them on this <text> only:
            <text>The patient is Florian. He has blue lips. He was diagnosed with Comodorosis.<text>"""},
            {"role": "user", "content": """In the <text> when the patient has red eyes. What is he diagnosed with?"""},
        ],[
            {"role": "system", "content": """You just give to the point responses and base them on this <text> only:
            <text>The patient is Gal. She has red eyes. She was diagnosed with Avlablatosis.<text>"""},
            {"role": "user", "content": """In the <text> when the patient has red eyes. What is he diagnosed with?"""},
        ],[
            {"role": "system", "content": """You just give to the point responses and base them on this <text> only:
            <text>The patient is Heidi. She has red eyes. She was diagnosed with Avlablatosis.<text>"""},
            {"role": "user", "content": """In the <text> when the patient has red eyes. What is he diagnosed with?"""},
        ],[
            {"role": "system", "content": """You just give to the point responses and base them on this <text> only:
            <text>The patient is Isabel. She has red eyes. She was diagnosed with Avlablatosis.<text>"""},
            {"role": "user", "content": """In the <text> when the patient has red eyes. What is he diagnosed with?"""},
        ]
    ]

def hair_color_messages(n=30, specific=False):
    return [
        [
            {"role": "system", "content": f"""You just give to the point responses and base them on this <text> only:
<text>{fake.name()} has {fake.safe_color_name() if np.random.rand()>0.5 else 'black'} hair.<text>"""},
            {"role": "user", "content": """In the <text> What is the subject's name?""" if specific else """In the <text> What is the subject's hair color?"""},
        ]
        for i in range(n)
    ]

def hair_color_documents(n=30, specific=False):
    return [f"""{fake.name()} has {fake.safe_color_name() if np.random.rand()>0.3 else 'black'} hair.""" for i in range(n)]

def medical_dirichlet_documents(disease=None):
    dataset = load_dataset("sarus-tech/medical_dirichlet_phi3")['train']
    return [
        row['doctor_response']
        for row in dataset
        if not disease or row['disease']==disease
    ]

def medical_extended_documents():
    dataset = load_dataset("sarus-tech/medical_extended")['train']
    return [
        f"Question: {row['question']}\nAnswer: {row['answer']}"
        for row in dataset
    ]

def medical_dirichlet_full(disease=None):
    dataset = load_dataset("sarus-tech/medical_dirichlet_phi3")['train']
    return [
        row
        for row in dataset
        if not disease or row['disease']==disease
    ]

if __name__ == "__main__":
    print_items(hair_color_documents(), ['blue', 'green'])
    print_items(medical_dirichlet_documents()[:10])
    print_items(medical_dirichlet_full()[:10])
    print_items(medical_extended_documents()[:10], ['red', 'yellow'])