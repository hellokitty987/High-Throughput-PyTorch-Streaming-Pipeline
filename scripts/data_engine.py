import webdataset as wds
import torch

"""
Builds a streaming loader that pulls data sequentially from shards
"""
def get_streaming_loader(shards_path, batch_size):

    # Define the stream from the .tar files
    dataset = (
        wds.WebDataset(shards_path)
        .decode("torch")
        .rename(video="pth", meta="json")
        .to_tuple("video", "meta")     
    )

    # Create the DataLoader
    loader = torch.utils.data.DataLoader (
        dataset=dataset,
        batch_size=batch_size,
        num_workers=0
    )

    return loader
