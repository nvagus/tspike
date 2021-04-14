import numpy as np
from hungarian_algorithm import algorithm


class AutoMatchingMatrix:
    def __init__(self, n_labels, n_outputs=None):
        self.n_labels = n_labels
        self.n_outputs = n_outputs or n_labels
        assert self.n_labels <= self.n_outputs, 'all labels should at least has a corresponding output'
        self.mat = np.zeros((self.n_labels, self.n_outputs))
    
    def add_sample(self, label, output):
        self.mat[label, output] += 1
    
    def reset(self):
        self.mat = np.zeros((self.n_labels, self.n_outputs))
    
    def describe(self):
        G = {
            f'label-{label}': {
                f'output-{output}': self.mat[label, output]
                for output in range(self.n_outputs)
            }
            for label in range(self.n_labels)
        }

        matching = algorithm.find_matching(G, matching_type='max', return_type='list')
        if matching:
            matching = {
                int(output.split('-')[1]): int(label.split('-')[1])
                for (label, output), _ in matching
            }
        else:
            matching = {}

        tp = np.zeros(self.n_labels)
        fp = np.zeros(self.n_labels)

        for output in range(self.n_outputs):
            label = matching[output] if output in matching else self.mat[:, output].argmax()
            tp[label] += self.mat[label, output]
            fp[label] += self.mat[:, output].sum() - self.mat[label, output]
        
        fn = self.mat.sum(axis=1) - tp
        
        accuracy = tp.sum() / self.mat.sum()
        # micro precision and recall
        precision = (tp / (tp + fp)).mean()
        recall = (tp / (tp + fn)).mean()
        return accuracy, precision, recall
    
    def describe_print_clear(self):
        print('Accuracy: {:.4f}; Precision: {:.4f}; Recall: {:.4f}'.format(*self.describe()))
