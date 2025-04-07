import json
import bisect
from collections import Counter
from termcolor import cprint
import matplotlib.pyplot as plt

class Analysis:
    def __init__(self, evaluation_path: str = 'results/evaluation.json'):
        with open(evaluation_path, 'r') as f:
            evaluation = json.load(f)
        self.evaluation = {}
        for key_str in evaluation:
            key_obj = json.loads(key_str)
            if isinstance(key_obj, list):
                self.evaluation[tuple(key_obj)] = evaluation[key_str]
    
    def disease_counts(self) -> list[tuple[str, int]]:
        result = []
        for key in self.evaluation:
            if len(key) == 2 and key[0] == 'symptoms' and key[1] != '*':
                result.append((key[1], self.evaluation[key]))
        return result
    
    def buckets(self) -> tuple[list[int], list[str]]:
        buckets = [10, 20, 50, 100, 500]
        labels = ['[0 10[','[10 20[','[20 50[','[50 100[','≥100']
        return buckets, labels
    
    def buckets_evaluation(self) -> dict[str, float]:
        buckets, labels = self.buckets()
        counts = Counter()
        successes = Counter()
        accuracies = Counter()
        for disease, count in self.disease_counts():
            index = bisect.bisect_right(buckets, count)
            counts[labels[index]] += self.evaluation[('symptoms', disease)]
            successes[labels[index]] += self.evaluation[('symptoms', disease, 5.3, 'success')]
            # counts[labels[index]] += 1
            # successes[labels[index]] += self.evaluation[('symptoms', disease, 5.3, 'success')]/self.evaluation[('symptoms', disease)]
        for label in labels:
            # if label != '≥200':
                accuracies[label] = successes[label]/counts[label]
        return accuracies



analysis = Analysis()
cprint(analysis.disease_counts(), 'grey')
accuracies = analysis.buckets_evaluation()
print(accuracies)
plt.figure(figsize=(10,8))
plt.title("Disease prediction accuracy as a function of disease frequency in the documents")
plt.xlabel("Disease occurences in the documents")
plt.ylabel("Disease prediction accuracy")
plt.plot(accuracies.keys(), accuracies.values(), linewidth=2, marker='o')
plt.fill_between(accuracies.keys(), accuracies.values(), alpha=0.2)
plt.savefig('images/accuracy.svg')


