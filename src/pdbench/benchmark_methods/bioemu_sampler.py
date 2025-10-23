import os
from src.pdbench.utils.sampler import BaseSampler
from bioemu.sample import main as sample


class BioemuSampler(BaseSampler):
    def __init__(self,
                 fasta_file_path: str,
                 sample_size: int,
                 save_path: str,
                 **kwargs) -> None:
        super().__init__(fasta_file_path,
                         sample_size,
                         save_path)

    def sample_single(self,
                      k: str,
                      sequence: str
                      ):
        sample_save_path = os.path.join(self.save_path, k.split(" ")[0])
        if not os.path.exists(sample_save_path):
            os.mkdir(sample_save_path)

        sample(sequence=sequence,
               num_samples=self.sample_size,
               output_dir=sample_save_path)

