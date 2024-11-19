import yaml
import os

class ParticleData:
    def __init__(self, names: list[str]) -> None:
        self.names = names
        self.type: dict[str, str] = {}
        self.mass: dict[str, float] = {}
        self.baryon_number: dict[str, int] = {}
        self.degeneracy: dict[str, int] = {}
        self.excluded_volume: dict[str, float] = {}
        self._load_data()

    def _load_data(self):
        file = os.getcwd() + '/eos/particle_data.yaml'
        stream = open(file, 'r')
        particle_data: dict = yaml.load(stream, Loader=yaml.FullLoader)
        for name in self.names:
            self.type[name] = particle_data[name]['type']
            self.mass[name] = particle_data[name]['mass']
            self.baryon_number[name] = particle_data[name]['baryon_number']
            self.degeneracy[name] = particle_data[name]['degeneracy']
            self.excluded_volume[name] = particle_data[name]['excluded_volume']

    def data(self, name: str) -> dict[str, any]:
        data: dict[str, any] = {}
        data['type'] = self.type[name]
        data['mass'] = self.mass[name]
        data['baryon_number'] = self.baryon_number[name]
        data['degeneracy'] = self.degeneracy[name]
        data['excluded_volume'] = self.excluded_volume[name]
        return data


if __name__ == '__main__':
    names = ['pi_zero', 'neutron', 'antineutron']
    pd = ParticleData(names)
