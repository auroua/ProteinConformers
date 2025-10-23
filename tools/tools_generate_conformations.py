import argparse
import torch


from src.pdbench.builder import get_sampler


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fasta_file_path", type=str)
    parser.add_argument("--sample_size", type=int, default=10000)
    parser.add_argument("--save_path", type=str)
    parser.add_argument("--sampler_type", type=str)
    parser.add_argument("--ckpt_path", type=str)
    parser.add_argument("--sample_steps", type=int)
    parser.add_argument("--sample_mode", type=str)
    parser.add_argument("--model_config_path", type=str)
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    sampler = get_sampler(args.sampler_type)(
        fasta_file_path = args.fasta_file_path,
        sample_size = args.sample_size,
        save_path = args.save_path,
        ckpt_path = args.ckpt_path,
        sample_steps = args.sample_steps,
        sample_mode = args.sample_mode,
        device = device,
        model_config_path=args.model_config_path
    )
    sampler.generation_conformation()