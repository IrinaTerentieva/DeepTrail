from pathlib import Path
from hydra.utils import get_original_cwd
from transformers import SegformerConfig, SegformerForSemanticSegmentation
import torch

def load_segformer(cfg):
    # Construct the absolute path to the local weights if needed
    if cfg.source == "huggingface":
        print(f"[INFO] Loading model from Hugging Face: {cfg.hf_repo_id}")
        model = SegformerForSemanticSegmentation.from_pretrained(cfg.hf_repo_id)
    else:
        # Build absolute path from original Hydra cwd + local path
        local_model_path = Path(get_original_cwd()) / cfg.local_path
        print(f"[INFO] Loading local weights from: {local_model_path}")

        config = SegformerConfig(
            num_labels=1,
            depths=cfg.depths,
            hidden_sizes=cfg.hidden_sizes,
            decoder_hidden_size=cfg.decoder_hidden_size,
            aspp_out_channels=cfg.get("aspp_out_channels", 256),
            aspp_ratios=cfg.get("aspp_ratios", [1, 6, 12, 18]),
            attention_probs_dropout_prob=cfg.get("attention_dropout_prob", 0.1),
        )
        model = SegformerForSemanticSegmentation(config)

        # Optional: If your input is single channel, adapt patch embeddings
        model.segformer.encoder.patch_embeddings[0].proj = torch.nn.Conv2d(
            in_channels=1,
            out_channels=model.segformer.encoder.patch_embeddings[0].proj.out_channels,
            kernel_size=model.segformer.encoder.patch_embeddings[0].proj.kernel_size,
            stride=model.segformer.encoder.patch_embeddings[0].proj.stride,
            padding=model.segformer.encoder.patch_embeddings[0].proj.padding,
            bias=False
        )

        state_dict = torch.load(local_model_path, map_location="cpu")
        model.load_state_dict(state_dict, strict=False)

    model.eval()
    return model
