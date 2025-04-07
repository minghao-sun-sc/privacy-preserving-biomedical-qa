import os
import sys
import json
import csv
import re
import logging
from dataclasses import dataclass
import torch
from model import Model, Config

logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))

@dataclass
class Dataset:
    model: Model = Model("meta-llama/Llama-3.2-3B-Instruct")
    # model: Model = Model("mistralai/Mistral-7B-Instruct-v0.3")
    # model: Model = Model("microsoft/Phi-3.5-mini-instruct")
    num_patients: int = 100
    num_diseases: int = 10
    
    def patients(self):
        """Build a list of fictitious patients"""
        outputs = set()
        os.makedirs('cache', exist_ok=True)
        try:
            with open('cache/patients.json', 'r') as f:
                outputs.update(json.load(f))
        except:
            logger.warning("No cache yet")
        while len(outputs)<self.num_patients:
            logger.info(f"Number of outputs {len(outputs)}")
            gen_outputs = self.model.chat_completion([[
                {"role": "system", "content": "You are a minimalist assistant, giving minimal answers"},
                {"role": "user", "content": "Generate a random european person first and last name. Just give first and last name"},
            ]]*512, Config(max_new_tokens=128, temperature=1.5))
            formatted_gen_outputs = [re.sub(r"[^a-zA-ZÀ-ÿ ]", "", output) for output in gen_outputs]
            for formatted_gen_output in formatted_gen_outputs:
                if ' ' in formatted_gen_output and len(formatted_gen_output)<32:
                    outputs.add(formatted_gen_output)
            with open('cache/patients.json', 'w') as f:
                json.dump(sorted(list(outputs)[:self.num_patients]), f, ensure_ascii=False, separators=(',\n', ': '))
        return outputs
    
    def diseases(self):
        """Build a list of fictitious diseases"""
        outputs = set()
        os.makedirs('cache', exist_ok=True)
        try:
            with open('cache/diseases.json', 'r') as f:
                outputs.update(json.load(f))
        except:
            logger.warning("No cache yet")
        while len(outputs)<self.num_diseases:
            logger.info(f"Number of outputs {len(outputs)}")
            gen_outputs = self.model.chat_completion([[
                {"role": "system", "content": "You are a minimalist assistant, giving minimal answers"},
                {"role": "user", "content": 'Generate a random invented funny disease name, just return the name]'},
            ]]*32, Config(max_new_tokens=128, temperature=1.5))
            formatted_gen_outputs = [re.sub(r"[^a-zA-ZÀ-ÿ0-9 ]", "", output) for output in gen_outputs]
            for formatted_gen_output in formatted_gen_outputs:
                if not ("fun" in formatted_gen_output or "Fun" in formatted_gen_output):
                    outputs.add(formatted_gen_output)
            with open('cache/diseases.json', 'w') as f:
                json.dump(sorted(list(outputs)[:self.num_diseases]), f, ensure_ascii=False, separators=(',\n', ': '))
        return outputs
    
    def symptoms(self):
        """Add symptoms to the disease"""
        outputs = dict()
        os.makedirs('cache', exist_ok=True)
        try:
            with open('cache/symptoms.json', 'r') as f:
                outputs = json.load(f)
        except:
            logger.warning("No cache yet")
        while len(outputs)<self.num_diseases:
            diseases = [disease for disease in self.diseases() if not (disease in outputs)]
            logger.info(f"Number of outputs {len(outputs)}")
            gen_outputs = self.model.chat_completion([[
                {"role": "system", "content": 'You are given a simple "<disease>" name, and you invent and return a simple list of few symptoms: ["<symptom1>", "<symptom2>", etc.]. Just give the list, no more.'},
                {"role": "user", "content": f'"{disease}"'},
            ] for disease in diseases], Config(max_new_tokens=128, temperature=1.0))
            formatted_gen_outputs = [(disease, re.search(r'\[(".*",\s*)*".*"\]', output)) for disease, output in zip(diseases, gen_outputs)]
            for disease, fgo in formatted_gen_outputs:
                if fgo and not (disease in outputs):
                    try:
                        outputs[disease] = json.loads(fgo.group())
                    except:
                        logger.warning(f"Cannot parse {fgo.group()}")
                else:
                    logger.warning(disease)
            with open('cache/symptoms.json', 'w') as f:
                json.dump(outputs, f, ensure_ascii=False, separators=(',\n', ': '))
        return outputs

    def treatments(self):
        """Add treatment to the disease"""
        outputs = dict()
        os.makedirs('cache', exist_ok=True)
        try:
            with open('cache/treatments.json', 'r') as f:
                outputs = json.load(f)
        except:
            logger.warning("No cache yet")
        while len(outputs)<self.num_diseases:
            diseases = [disease for disease in self.diseases() if not (disease in outputs)]
            logger.info(f"Number of outputs {len(outputs)}")
            gen_outputs = self.model.chat_completion([[
                {"role": "system", "content": 'You are given a simple "<disease>" name, and you invent and return a simple fictitious - and funny - treatment name: <treatment>. Just give the treatment name, no more.'},
                {"role": "user", "content": f'"{disease}"'},
            ] for disease in diseases], Config(max_new_tokens=128, temperature=1.0))
            formatted_gen_outputs = [(disease, re.sub(r"[^a-zA-ZÀ-ÿ0-9 ]", "", output)) for disease, output in zip(diseases, gen_outputs)]
            for disease, fgo in formatted_gen_outputs:
                if fgo and not (disease in outputs):
                    outputs[disease] = fgo
                else:
                    logger.warning(disease)
            with open('cache/treatments.json', 'w') as f:
                json.dump(outputs, f, ensure_ascii=False, separators=(',\n', ': '))
        return outputs
    
    def disease_probabilities(self):
        outputs = dict()
        os.makedirs('cache', exist_ok=True)
        try:
            with open('cache/disease_probabilities.json', 'r') as f:
                outputs = json.load(f)
        except:
            logger.warning("No cache yet")
        while len(outputs)<self.num_diseases:
            diseases = [disease for disease in self.diseases() if not (disease in outputs)]
            probabilities = torch.distributions.Dirichlet(torch.ones(len(diseases))).sample().tolist()
            outputs.update(dict(zip(diseases, probabilities)))
            with open('cache/disease_probabilities.json', 'w') as f:
                json.dump(outputs, f, ensure_ascii=False, separators=(',\n', ': '))
        return outputs

    def patient_diseases(self):
        outputs = dict()
        os.makedirs('cache', exist_ok=True)
        try:
            with open('cache/patient_diseases.json', 'r') as f:
                outputs = json.load(f)
        except:
            logger.warning("No cache yet")
        while len(outputs)<self.num_diseases:
            patients = [patient for patient in self.patients() if not (patient in outputs)]
            disease_probabilities = self.disease_probabilities()
            diseases, probabilities = zip(*disease_probabilities.items())
            gen = torch.distributions.categorical.Categorical(torch.Tensor(probabilities))
            outputs.update({patient: diseases[index] for patient, index in zip(patients, gen.sample((len(patients),)).tolist())})
            with open('cache/patient_diseases.json', 'w') as f:
                json.dump(outputs, f, ensure_ascii=False, separators=(',\n', ': '))
        return outputs

    def patient_requests(self):
        """Generate patient requests"""
        outputs = dict()
        os.makedirs('cache', exist_ok=True)
        try:
            with open('cache/patient_requests.json', 'r') as f:
                outputs = json.load(f)
        except:
            logger.warning("No cache yet")
        while len(outputs)<self.num_patients:
            patients = [patient for patient in self.patients() if not (patient in outputs)][:128]#[:1024]
            patient_diseases = self.patient_diseases()
            patient_disease_items = [(patient, patient_diseases[patient]) for patient in patients]
            logger.info(f"Number of outputs {len(outputs)}")
            gen_outputs = self.model.chat_completion([[
                {"role": "system", "content": 'You reformulate the user sentence.'},
                {"role": "user", "content": f'I am {patient}, I experience {", ".join(self.symptoms()[disease])}'},
            ] for patient, disease in patient_disease_items], Config(max_new_tokens=128, temperature=1.0))
            formatted_gen_outputs = [(patient, output) for patient, output in zip(patients, gen_outputs)]
            for patient, fgo in formatted_gen_outputs:
                if not (patient in outputs):
                    outputs[patient] = fgo
                else:
                    logger.warning(patient)
            with open('cache/patient_requests.json', 'w') as f:
                json.dump(outputs, f, ensure_ascii=False, separators=(',\n', ': '))
        return outputs
    
    def doctor_responses(self):
        """Generate doctor responses"""
        outputs = dict()
        os.makedirs('cache', exist_ok=True)
        try:
            with open('cache/doctor_responses.json', 'r') as f:
                outputs = json.load(f)
        except:
            logger.warning("No cache yet")
        while len(outputs)<self.num_patients:
            patients = [patient for patient in self.patients() if not (patient in outputs)][:128]#[:1024]
            patient_diseases = self.patient_diseases()
            patient_disease_items = [(patient, patient_diseases[patient]) for patient in patients]
            logger.info(f"Number of outputs {len(outputs)}")
            gen_outputs = self.model.chat_completion([[
                {"role": "system", "content": 'You reformulate the user sentence.'},
                {"role": "user", "content": f'Given the symptoms ({", ".join(self.symptoms()[disease])}) of {patient}. The diagnosis is {disease}, the treatment should be {self.treatments()[disease]}.'},
            ] for patient, disease in patient_disease_items], Config(max_new_tokens=128, temperature=1.0))
            formatted_gen_outputs = [(patient, output) for patient, output in zip(patients, gen_outputs)]
            for patient, fgo in formatted_gen_outputs:
                if not (patient in outputs):
                    outputs[patient] = fgo
                else:
                    logger.warning(patient)
            with open('cache/doctor_responses.json', 'w') as f:
                json.dump(outputs, f, ensure_ascii=False, separators=(',\n', ': '))
        return outputs
    
    def rows(self):
        patient_requests = self.patient_requests()
        doctor_responses = self.doctor_responses()
        patient_diseases = self.patient_diseases()
        symptoms = self.symptoms()
        treatments = self.treatments()
        for id, patient in enumerate(self.patients()):
            patient_request = patient_requests[patient]
            doctor_response = doctor_responses[patient]
            disease = patient_diseases[patient]
            symptom = symptoms[disease]
            treatment = treatments[disease]
            yield {
                "patient_id": id,
                "patient_name": patient,
                "patient_request": patient_request,
                "doctor_response": doctor_response,
                "symptom": symptom,
                "disease": disease,
                "treatment": treatment,
            }

    def csv(self, path: str='medical_data.csv'):
        row = next(self.rows())
        with open(path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())
            writer.writeheader()
            for row in self.rows():
                writer.writerow(row)
    
    def jsonl(self, path: str='medical_data.jsonl'):
        with open(path, 'w') as f:
            for row in self.rows():
                f.write(f"{json.dumps(row)}\n")

def main():
    dataset = Dataset(num_patients=10000, num_diseases=100)
    dataset.jsonl()


if __name__ == "__main__":
    main()