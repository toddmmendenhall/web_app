import numpy as np


class PhaseDiagram:
    def __init__(self, bounds_and_steps: list[float]) -> None:
        self._temp_min = bounds_and_steps[0]
        self._temp_step = bounds_and_steps[2]
        self._temp_max = bounds_and_steps[1] + self._temp_step
        self._muB_min = bounds_and_steps[3]
        self._muB_step = bounds_and_steps[5]
        self._muB_max = bounds_and_steps[4] + self._muB_step

        self.temp = np.arange(self._temp_min, self._temp_max, self._temp_step)
        self.muB = np.arange(self._muB_min, self._muB_max, self._muB_step)
    
    def grid(self) -> tuple[np.ndarray[np.float64]]:
        return np.meshgrid(self.temp, self.muB)
    
    def shape(self) -> list[int]:
        return [self.muB.size, self.temp.size]

if __name__ == '__main__':
    phase_diagram = PhaseDiagram([0, 200, 50, 0, 1000, 50])
    print(phase_diagram.shape())