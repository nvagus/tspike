import torch
import numpy as np
from hungarian_algorithm import algorithm
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score

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
                f'output-{output}': self.mat[label, output] + 1
                for output in range(self.n_outputs)
            }
            for label in range(self.n_labels)
        }

        matching = algorithm.find_matching(
            G, matching_type='max', return_type='list')
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
            label = matching[output] if output in matching else self.mat[:, output].argmax(
            )
            tp[label] += self.mat[label, output]
            fp[label] += self.mat[:, output].sum() - self.mat[label, output]

        fn = self.mat.sum(axis=1) - tp

        accuracy = tp.sum() / self.mat.sum()
        # micro precision and recall
        precision = (tp / (tp + fp)).mean()
        recall = (tp / (tp + fn)).mean()
        return accuracy, precision, recall

    def describe_print_clear(self):
        print('Accuracy: {:.4f}; Precision: {:.4f}; Recall: {:.4f}'.format(
            *self.describe()))


class SpikesTracer:
    def __init__(self, dummy_label=10):
        self.real_labels = []
        self.predicted_labels = []

        self.dummy_label = dummy_label

    def describe_batch_spikes(self, output_spikes):
        # output_spikes - batch, output_channel, neuro, time

        batch_summed_output = output_spikes.sum(0)

        # has spikes
        has_spikes = torch.any(batch_summed_output).item()

        # spike time
        spike_time = batch_summed_output.sum(
            (0, 1)).nonzero().float().squeeze(-1)
        avg_spike_time = torch.mean(spike_time).item()
        last_spike_time = torch.max(spike_time).item()

        # total_number_of_spikes per channel
        channel_summed_spikes = batch_summed_output.sum((1, 2))

        # argmax spike channel
        time_summed_spikes = output_spikes.sum(
            3)  # batch, output_channel, neuro
        channel_argmax = torch.argmax(
            time_summed_spikes, 1).squeeze(-1)  # batch, neuro

        result = {}
        result["has_spike"] = has_spikes
        result["avg_spike_time"] = avg_spike_time
        result["last_spike_time"] = last_spike_time
        result["channel_summed_spikes"] = channel_summed_spikes.cpu().numpy()
        result["channel_argmax"] = channel_argmax.cpu().numpy()

        return result

    def get_predict(self, output_spikes):
        y_spikes = output_spikes.sum((-2, -1))  # batch, output_channel
        y_preds = y_spikes.argmax(-1)  # select argmax as label
        y_preds[y_spikes.sum(1) == 0] = self.dummy_label

        y_preds = y_preds.cpu().numpy()

        return y_preds

    # evaluation
    # accuracy (including the no-spikes), percision, recall
    # spike probability
    # confusion matrix (10 * 11)
    def add_sample(self, label, output):
        self.real_labels += list(label)
        self.predicted_labels += list(output)

    def evaluate(self, clean_state=False):
        accuracy = accuracy_score(self.real_labels, self.predicted_labels)
        percision = precision_score(
            self.real_labels, self.predicted_labels, average='micro')
        recall = recall_score(
            self.real_labels, self.predicted_labels, average='micro')

        matrix = confusion_matrix(self.real_labels, self.predicted_labels)

        if clean_state:
            self.clear()

        return accuracy, percision, recall, matrix

    def clear(self):
        self.real_labels = []
        self.predicted_labels = []

    def describe_print_clear(self):
        accuracy, percision, recall, matrix = self.evaluate()
        print('Accuracy: {:.4f}; Precision: {:.4f}; Recall: {:.4f}'.format(
            accuracy, percision, recall))
        print(matrix)
