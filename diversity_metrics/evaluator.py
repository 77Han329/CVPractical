import os
from PIL import Image
from tqdm import tqdm

class MetricEvaluator:
    def __init__(self, metrics):
        self.metrics = metrics  # list of metric instances

    def compute(self, folder1, folder2, save_dir=None):
        files1 = sorted(os.listdir(folder1))
        files2 = sorted(os.listdir(folder2))

        assert len(files1) == len(files2), "Folders must contain the same amount of images."

        results = {metric.name(): {} for metric in self.metrics}

        for file1, file2 in tqdm(zip(files1, files2), total=len(files1), desc="Computing Metrics"):
            path1 = os.path.join(folder1, file1)
            path2 = os.path.join(folder2, file2)

            if not os.path.exists(path1) or not os.path.exists(path2):
                continue

            img1 = Image.open(path1).convert('RGB')
            img2 = Image.open(path2).convert('RGB')

            for metric in self.metrics:
                score = metric.compute_distance(img1, img2)
                results[metric.name()][file1] = score

        # optionally save
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            for name, scores in results.items():
                with open(os.path.join(save_dir, f"{name}.txt"), "w") as f:
                    for file, score in scores.items():
                        f.write(f"{file}: {score:.6f}\n")

        # compute averages
        avg_scores = {
            name: sum(scores.values()) / (len(scores) + 1e-8)
            for name, scores in results.items()
        }

        return avg_scores
