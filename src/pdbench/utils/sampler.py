from abc import ABC, abstractmethod
from tqdm import tqdm
from src.pdbench.utils.data_utils import parse_fasta


class BaseSampler(ABC):
    def __init__(self,
                 fasta_file_path: str,
                 sample_size: int,
                 save_path: str,
                 **kwargs) -> None:
        self.fasta_file_path = fasta_file_path
        self.sample_size = sample_size
        self.fasta_data = parse_fasta(fasta_file_path)
        self.fasta_data_keys = [k for k in self.fasta_data.keys()]
        self.save_path = save_path

    def get_fasta_data(self,
                       idx: int):
        return self.fasta_data[self.fasta_data_keys[idx]]

    def __len__(self):
        return len(self.fasta_data)

    @abstractmethod
    def sample_single(self,
                      k: str,
                      sequence: str):
        pass

    def generation_conformation(self):
        for k, v in tqdm(self.fasta_data.items()):
            print(f"Start generating conformations for {k.split(' ')[0]}, and the sequence is {v}")
            self.sample_single(k, v)