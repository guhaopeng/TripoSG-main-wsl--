import argparse
import os
import sys

import numpy as np
import torch
import trimesh

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from huggingface_hub import snapshot_download

from triposg.inference_utils import hierarchical_extract_geometry, flash_extract_geometry
from triposg.models.autoencoders import TripoSGVAEModel


DTYPE_MAP = {
    "float16": torch.float16,
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
}


def maybe_download_weights(weights_dir: str) -> str:
    if os.path.isdir(weights_dir) and os.path.isdir(os.path.join(weights_dir, "vae")):
        return weights_dir
    snapshot_download(repo_id="VAST-AI/TripoSG", local_dir=weights_dir)
    return weights_dir


def load_latent_tensor(
    latent_path: str,
    device: str,
    dtype: torch.dtype,
) -> torch.Tensor:
    if not os.path.exists(latent_path):
        raise FileNotFoundError(f"Latent file not found: {latent_path}")

    latent = torch.load(latent_path, map_location="cpu")
    if isinstance(latent, dict):
        if "latent" in latent:
            latent = latent["latent"]
        elif "latents" in latent:
            latent = latent["latents"]
        else:
            raise ValueError(
                f"Unsupported latent file format: keys={list(latent.keys())}"
            )
    if not isinstance(latent, torch.Tensor):
        raise TypeError(f"Loaded latent must be a tensor, got {type(latent)}")

    if latent.ndim == 2:
        latent = latent.unsqueeze(0)
    if latent.ndim != 3:
        raise ValueError(
            f"Expected latent with shape [B, T, C] or [T, C], got {tuple(latent.shape)}"
        )

    return latent.to(device=device, dtype=dtype)


def export_random_latent_mesh(args):
    dtype = DTYPE_MAP[args.dtype]
    device = args.device

    weights_dir = maybe_download_weights(args.triposg_weights)
    vae: TripoSGVAEModel = TripoSGVAEModel.from_pretrained(
        weights_dir,
        subfolder="vae",
    ).to(device, dtype=dtype)
    vae.eval()

    latents = load_latent_tensor(args.latent_path, device=device, dtype=dtype)
    print(f"Loaded latent from {args.latent_path}, shape={tuple(latents.shape)}")

    bounds = (
        -1.005,
        -1.005,
        -1.005,
        1.005,
        1.005,
        1.005,
    )

    with torch.no_grad():
        if args.use_flash_decoder:
            vae.set_flash_decoder()
            output = flash_extract_geometry(
                latents,
                vae,
                bounds=bounds,
                num_chunks=args.flash_num_chunks,
                octree_depth=args.flash_octree_depth,
            )
        else:
            geometric_func = lambda x: vae.decode(
                latents,
                sampled_points=x,
                num_chunks=args.decode_num_chunks,
            ).sample
            output = hierarchical_extract_geometry(
                geometric_func,
                device,
                bounds=bounds,
                dense_octree_depth=args.dense_octree_depth,
                hierarchical_octree_depth=args.hierarchical_octree_depth,
            )

        meshes = [
            trimesh.Trimesh(vertices.astype(np.float32), faces)
            for vertices, faces in output
        ]

    output_dir = os.path.dirname(args.output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    if len(meshes) == 1:
        meshes[0].export(args.output_path)
        print(f"Saved mesh to {args.output_path}")
    else:
        base, ext = os.path.splitext(args.output_path)
        for i, mesh in enumerate(meshes):
            mesh_path = f"{base}_{i}{ext}"
            mesh.export(mesh_path)
            print(f"Saved mesh to {mesh_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Load an existing TripoSG latent and decode it into a mesh."
    )
    parser.add_argument(
        "--triposg_weights",
        type=str,
        default="pretrained_weights/TripoSG",
        help="Path to TripoSG pretrained weights. Auto-downloads if missing.",
    )
    parser.add_argument(
        "--latent_path",
        type=str,
        default="latetns.pt",
        help="Path to an existing latent tensor (.pt). Supports [T, C] or [B, T, C].",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="decoded_latent.glb",
        help="Output mesh path (.glb, .obj, .ply, etc.).",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--dtype",
        type=str,
        default="float16",
        choices=sorted(DTYPE_MAP.keys()),
    )
    parser.add_argument(
        "--decode_num_chunks",
        type=int,
        default=50000,
        help="Maximum number of query points per internal decode chunk.",
    )
    parser.add_argument("--use_flash_decoder", action="store_true")
    parser.add_argument("--dense_octree_depth", type=int, default=8)
    parser.add_argument("--hierarchical_octree_depth", type=int, default=9)
    parser.add_argument("--flash_octree_depth", type=int, default=8)
    parser.add_argument(
        "--flash_num_chunks",
        type=int,
        default=10000,
        help="Chunk budget used by flash geometry extraction.",
    )
    export_random_latent_mesh(parser.parse_args())
