from abc import ABC, abstractmethod

class Metric(ABC):
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def compute_distance(self, img1, img2) -> float:
        pass
