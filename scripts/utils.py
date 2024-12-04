from sklearn.metrics import precision_score, explained_variance_score, accuracy_score, classification_report, roc_auc_score, accuracy_score, log_loss, mean_squared_error, mean_absolute_error, r2_score, mean_squared_log_error
import matplotlib.pyplot as plt

class RegressionMetrics:
    def __init__(self):
        self.results = {}

    def run(self, y_true, y_pred, method_name):
        # Calculate regression metrics
        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)

        # Store results
        self.results[method_name] = {
            'MSE': mse,
            'MAE': mae
        }
    def plot(self):
        # Create subplots
        fig, axs = plt.subplots(2, figsize=(15, 10))

        # Plot each metric
        metrics = ['MSE', 'MAE']
        for i, metric in enumerate(metrics):
            ax = axs[i]  # Use a single index to access subplots in a 1D array
            values = [res[metric] for res in self.results.values()]
            ax.bar(self.results.keys(), values)
            ax.set_title(metric)

            # Add values on the bars
            for j, v in enumerate(values):
                ax.text(j, v + v * 0.05, f"{v:.2f}", ha='center', va='bottom')

        plt.tight_layout()
        plt.show()

class ClassificationMetrics:
    def __init__(self):
        self.results = {}

    def run(self, y_true, y_pred, method_name, average='macro'):
        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average=average)
        recall = recall_score(y_true, y_pred, average=average)
        f1 = f1_score(y_true, y_pred, average=average)

        # Store results
        self.results[method_name] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
        }

    def plot(self):
        # Create subplots
        fig, axs = plt.subplots(2, 2, figsize=(15, 10))

        # Plot each metric
        for i, metric in enumerate(['accuracy', 'precision', 'recall', 'f1']):
            ax = axs[i//2, i%2]
            values = [res[metric] * 100 for res in self.results.values()]
            ax.bar(self.results.keys(), values)
            ax.set_title(metric)
            ax.set_ylim(0, 100)

            # Add values on the bars
            for j, v in enumerate(values):
                ax.text(j, v + 0.02, f"{v:.2f}", ha='center', va='bottom')

        plt.tight_layout()
        plt.show()

