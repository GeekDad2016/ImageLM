import torch
from torchvision.transforms import ToTensor

from nanochat.common import get_dist_info
from nanochat.dataset import parquets_iter_batched
from nanochat.image_renderer import render_text_to_image

def image_generating_distributed_data_loader(B, T, split, device="cuda"):
    """Stream pretraining text from parquet files, render to images, yield training batches."""
    assert split in ["train", "val"], "split must be 'train' or 'val'"
    ddp, ddp_rank, ddp_local_rank, ddp_world_size = get_dist_info()

    def document_batches():
        while True:
            for batch in parquets_iter_batched(split=split, start=ddp_rank, step=ddp_world_size):
                yield batch
    batches = document_batches()

    while True:
        doc_batch = next(batches)
        for i in range(0, len(doc_batch), B):
            batch = doc_batch[i:i+B]
            images = [render_text_to_image(text) for text in batch]
            
            transform = ToTensor()
            tensors = [transform(image) for image in images]
            
            inputs = torch.stack(tensors).to(device=device, non_blocking=True)
            # Note: Targets will need to be handled differently now.
            # For now, let's just yield the inputs.
            yield inputs, None
        break
