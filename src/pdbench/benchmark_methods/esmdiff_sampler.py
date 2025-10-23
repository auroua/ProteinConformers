import os
import torch
from functools import partial
from pathlib import Path
from time import time, strftime
import tempfile
from esm.sdk.api import ESMProtein, GenerationConfig
from esm.utils.generation import iterative_sampling_raw
from esm.utils.constants import esm3 as C
from esm.sdk.api import ESMProtein, ESMProteinTensor
from esm.models.esm3 import ESM3, ESMOutput

from src.pdbench.utils.sampler import BaseSampler
from src.pdbench.third_party.esmdiff.slm.utils.checkpoint_utils import load_state_dict_from_lightning_ckpt
from src.pdbench.third_party.esmdiff.slm.utils.eval_utils import timer, merge_pdbfiles
from src.pdbench.third_party.esmdiff.slm.models.utils import protseq_to_data

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
_model_esm3 = ESM3.from_pretrained("esm3_sm_open_v1").to(device)


class ESMDiffSampler(BaseSampler):
    def __init__(self,
                 fasta_file_path: str,
                 sample_size: int,
                 save_path: str,
                 ckpt_path: str,
                 sample_steps: int,
                 sample_mode: str,
                 device: torch.device,
                 model_config_path: str,
                 **kwargs) -> None:
        super().__init__(fasta_file_path,
                         sample_size,
                         save_path)
        self.ckpt_path = ckpt_path
        self.sample_steps = sample_steps
        self.sample_mode = sample_mode
        self.device = device
        self.model_config_path = model_config_path

        self.model = load_state_dict_from_lightning_ckpt(self.ckpt_path, self.device, exp_cfg_path=self.model_config_path)
        self.net = self.model.net

        self.sample_fn = {
            "gibbs": partial(minibatch_gibbs_by_esm, esm3_model=self.net),
            "ddpm": partial(ddpm_sample_by_esm, pl_model=self.model),
        }[self.sample_mode]

        print(f">>> Sampling mode = {self.sample_mode} ...")
        ##############################################################################
        self.output_dir = Path(self.save_path)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def sample_single(self,
                      k: str,
                      sequence: str
                      ):
        sample_save_path = os.path.join(self.save_path, k.split(" ")[0])
        os.mkdir(sample_save_path)
        # prot = ESMProtein(sequence=sequence)
        coordinates = None
        self.sample_fn(
            k,
            sequence,
            output_dir=self.output_dir,
            sample_basename=k,
            num_samples=self.sample_size,
            num_steps=self.sample_steps,
            coordinates=coordinates,
            mask_ids=None,
        )


@timer
@torch.no_grad()
def minibatch_gibbs_by_esm(
        key,
        sequence,
        protseq,
        esm3_model,
        output_dir: Path,
        sample_basename: str,
        num_samples: int = 10,
        num_steps: int = 16,
        temperature: float = 1.4,
        top_p: float = 0.9,
        n_max_residue_square: int = 200 * 200 * 105,
        coordinates=None,
        mask_ids=None,
):

    str_time = strftime("%Y%m%d-%H%M%S")
    output_dir = output_dir / f"{key}_T{temperature}_step{num_steps}_topp{top_p}_N{num_samples}_{str_time}"
    save_to = output_dir / f"{sample_basename}.pdb"
    print(f"Results will save to {save_to}")
    if save_to.exists():
        print(f"Skip existing {save_to}")
        return None

    if mask_ids is not None:
        print(f"Masking {len(mask_ids)} residues and inpainting...")
        assert coordinates is not None, "Need to provide coordinates for masking"
        protseq = list(protseq)
        for idx in mask_ids:
            assert 0 <= idx < len(sequence), f"Invalid mask index {idx} for sequence of length {len(sequence)}"
            protseq[idx] = '_'
            coordinates[idx] = float('Inf')
        protseq = ''.join(protseq)

    output_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    with tempfile.TemporaryDirectory() as tmpdirname:
        out_list = []
        batchs = []
        bsz = []
        target_size = len(protseq) * len(protseq) * num_samples
        n_batch = target_size // n_max_residue_square
        residual_size = target_size % n_max_residue_square
        batch_size = n_max_residue_square // int(len(protseq) * len(protseq))
        for i in range(n_batch):
            bsz.append(batch_size)
        if residual_size > 0:
            bsz.append(num_samples - sum(bsz))
        assert sum(bsz) == num_samples, f"{sum(bsz)} != {num_samples}"
        for bs in bsz:
            prot_list = [ESMProtein(sequence=protseq, coordinates=coordinates) for _ in range(bs)]
            print(f"Generating {len(prot_list)} samples for {protseq}...")
            cfg_list = [
                GenerationConfig(track="structure", num_steps=num_steps, temperature=temperature, top_p=top_p)
                for _ in range(bs)
            ]
            out_list += iterative_sampling_raw(
                esm3_model, proteins=prot_list, configs=cfg_list,
            )  # input/output have the same type
        out_list = [out for out in out_list if isinstance(out, ESMProtein)]
        for i, prot in enumerate(out_list):
            base = sample_basename + '.' + f"{i}.pdb"
            tmp = Path(tmpdirname) / base
            prot.to_pdb(tmp)
            saved.append(tmp)
        merge_pdbfiles(saved, save_to, verbose=False)
    return out_list


# esm=3.0.2
# batch generation
@timer
@torch.no_grad()
def ddpm_sample_by_esm(
        key,
        sequence,
        pl_model,
        output_dir: Path,
        sample_basename: str,
        esm3_model: ESM3 = _model_esm3,
        num_samples: int = 5,
        num_steps: int = 10,
        eps: float = 1e-5,
        n_max_residue_square: int = 200 * 200 * 105,
        coordinates=None,
        mask_ids=None,
        filled_ids=None,
        total_size=None,
        sample_max_t: float = 1.0,
):
    model = pl_model
    str_time = strftime("%Y%m%d-%H%M%S")
    output_dir = output_dir / f"{sample_basename}_step{num_steps}_eps{eps}_N{num_samples}_{str_time}"
    save_to = output_dir / f"{sample_basename}.pdb"
    print(f"Results will save to {save_to}")
    if save_to.exists():
        print(f"Skip existing {save_to}")
        return None

    output_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    counter = 0
    with tempfile.TemporaryDirectory() as tmpdirname:
        # prot_list = [ESMProtein(sequence=protseq) for _ in range(num_samples)]
        prot = protseq_to_data(
            sequence,
            esm3_model,
            encode_only=True,
            coordinates=coordinates,
            mask_ids=mask_ids,
            filled_ids=filled_ids,
            total_size=total_size,
        )
        sequence_tokens_singleton = prot["sequence_tokens"]
        esm3_model = esm3_model.to('cpu')
        start_t = time()
        structure_tokens_list = []
        batchs = []
        bsz = []
        target_size = sequence_tokens_singleton.size(0) * sequence_tokens_singleton.size(0) * num_samples
        n_batch = target_size // n_max_residue_square
        residual_size = target_size % n_max_residue_square
        batch_size = n_max_residue_square // int(sequence_tokens_singleton.size(0) * sequence_tokens_singleton.size(0))
        for i in range(n_batch):
            batchs.append(sequence_tokens_singleton[None, :].repeat(batch_size, 1))
            bsz.append(batch_size)

        if residual_size > 0:
            batchs.append(sequence_tokens_singleton[None, :].repeat(num_samples - sum(bsz), 1))
            bsz.append(num_samples - sum(bsz))

        assert sum(bsz) == num_samples, f"{sum(bsz)} != {num_samples}"
        print(f"Total {num_samples} samples will be generated in batchs {bsz}...")

        sequence_tokens_singleton = sequence_tokens_singleton[1:-1]

        for batch in batchs:
            if mask_ids is not None:
                # batch for parallel sampling
                input_prior = prot['structure_tokens'][None, :].repeat(batch.size(0), 1)
                for idx in mask_ids:  # to generate these residues (unknown residues)
                    input_prior[:, idx] = C.STRUCTURE_MASK_TOKEN
            elif filled_ids is not None:
                # batch for parallel sampling
                input_prior = prot['structure_tokens'][None, :].repeat(batch.size(0), 1)
                for idx in range(total_size):  # to exclude these residues during generation (known residues)
                    if idx not in filled_ids:
                        input_prior[:, idx] = C.STRUCTURE_MASK_TOKEN
            else:
                input_prior = None
            structure_tokens_list.append(model.ddpm_sample(
                num_steps=num_steps,
                sequence_tokens=batch,
                eps=eps,
                input_prior=input_prior,
                sample_max_t=sample_max_t,
            ))
            structure_tokens = torch.cat(structure_tokens_list, dim=0)

            # remove bos and eos positions
            structure_tokens = structure_tokens[:, 1:-1]

            print(f"Sampling token time: {time() - start_t:.2f}s")
            out_list = []
            for i in range(len(structure_tokens)):
                base = sample_basename + '.' + f"{counter}.pdb"
                counter += 1
                tmp = Path(tmpdirname) / base
                tmp_output = Path(output_dir) / base
                st_i = structure_tokens[i]
                decode(structure_tokens=st_i, sequence_tokens=sequence_tokens_singleton, esm3_model=esm3_model, save_to=tmp)
                decode(structure_tokens=st_i, sequence_tokens=sequence_tokens_singleton, esm3_model=esm3_model, save_to=tmp_output)
                saved.append(tmp)
            structure_tokens_list = []
        merge_pdbfiles(saved, save_to, verbose=False)
        print(f"Total time: {time() - start_t:.2f}s")
    return out_list


@torch.no_grad()
def decode(structure_tokens, sequence_tokens, esm3_model, save_to=None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # per-sample input!!
    assert len(structure_tokens) == len(sequence_tokens), f"{len(structure_tokens)} != {len(sequence_tokens)}"
    # add BOS and EOS to tensors
    sequence_tokens = torch.cat(
        [torch.LongTensor([C.SEQUENCE_BOS_TOKEN]),
         sequence_tokens.cpu(),
         torch.LongTensor([C.SEQUENCE_EOS_TOKEN])]
    )
    structure_tokens = torch.cat(
        [torch.LongTensor([C.STRUCTURE_BOS_TOKEN]),
         structure_tokens.cpu(),
         torch.LongTensor([C.STRUCTURE_EOS_TOKEN])]
    )

    prot = ESMProteinTensor(sequence=sequence_tokens, structure=structure_tokens)
    prot = prot.to(device)
    esm3_model = esm3_model.to(device)
    raw_protein = esm3_model.decode(prot)
    if save_to is not None:
        raw_protein.to_pdb(save_to)