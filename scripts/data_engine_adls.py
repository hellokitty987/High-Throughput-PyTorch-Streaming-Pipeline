
import re
from typing import Iterable, List, Union
import torch
import webdataset as wds

try:
    import fsspec
except ImportError as exc:
    raise ImportError(
        "This cloud-aware loader requires fsspec. Install with: pip install fsspec adlfs"
    ) from exc

AZURE_STORAGE_OPTIONS = {
    # "account_name": "stpytorchdatalakedev",
    # "tenant_id": os.getenv("AZURE_TENANT_ID"),
    # "client_id": os.getenv("AZURE_CLIENT_ID"),
    # "client_secret": os.getenv("AZURE_CLIENT_SECRET"),
}


def expand_shard_pattern(shards_path: Union[str, Iterable[str]]) -> List[str]:

    if not isinstance(shards_path, str):
        return list(shards_path)

    match = re.search(r"\{(\d+)\.\.(\d+)\}", shards_path)
    if not match:
        return [shards_path]

    start_text, end_text = match.groups()
    width = len(start_text)
    start = int(start_text)
    end = int(end_text)

    shard_urls = []
    for shard_number in range(start, end + 1):
        replacement = f"{shard_number:0{width}d}"
        shard_urls.append(shards_path[: match.start()] + replacement + shards_path[match.end() :])

    return shard_urls


def fsspec_url_opener(data, handler=wds.reraise_exception):

    for sample in data:
        url = sample["url"]

        try:
            stream = fsspec.open(url, "rb", **AZURE_STORAGE_OPTIONS).open()
            sample.update(stream=stream)
            yield sample
        except Exception as exception:
            if handler(exception):
                continue
            break


def get_streaming_loader(shards_path, batch_size):
    shard_urls = expand_shard_pattern(shards_path)

    dataset = wds.DataPipeline(
        wds.SimpleShardList(shard_urls),
        fsspec_url_opener,
        wds.tar_file_expander,
        wds.group_by_keys,
        wds.decode("torch"),
        wds.rename(video="pth", meta="json"),
        wds.to_tuple("video", "meta"),
    )

    loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=0,
    )

    return loader
