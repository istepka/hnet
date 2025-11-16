import os
import shutil

import numpy as np
from datasets import load_dataset

from hnet.utils.tokenizers import (
    ByteTokenizer,
    TSContinuousTokenizer,
    TSQuantizedTokenizer,
)


def prepare_text_tokenized_data(
    tokenizer: ByteTokenizer,
    tokens_path: str,
    hf_cache_path: str,
    regenerate: bool = False,
    path: str = "Salesforce/wikitext",
    name: str = "wikitext-103-v1",
    split: str = "train",
    streaming: bool = False,
    take_rows: int | None = None,
    chunk_toklimit: int = 250_000_000,  # 1GB per file
    **kwargs,
) -> str:
    """
    Will prepare tokenized data if not already present.
    Returns path to tokenized data.

    For now we merge all data into single binary file. For simplicity.
    """
    if not os.path.exists(tokens_path) or regenerate:
        if regenerate:
            if os.path.exists(tokens_path):
                print("Removing old data")
                shutil.rmtree(tokens_path)

        print("Tokenized data not found, creating...")
        os.makedirs(tokens_path)

        dataset = load_dataset(
            path, name=name, streaming=streaming, split=split, cache_dir=hf_cache_path
        )

        if streaming:
            cnt = 0
            chunks = 0
            texts = []
            for i, doc in enumerate(dataset.take(take_rows)):
                texts.append(doc["text"])
                cnt += len(doc["text"])
                if cnt >= chunk_toklimit or i == (take_rows - 1):
                    encoded_inputs = tokenizer.encode(texts, add_bos=True, add_eos=True)

                    longs_ids = list()
                    for ids in encoded_inputs:
                        longs_ids.append(np.array(ids["input_ids"], dtype=np.int32))
                    print(f"Saving tokenized data chunk no. {chunks}")
                    np.concatenate(longs_ids).tofile(
                        os.path.join(tokens_path, f"tokens_{chunks}.bin")
                    )
                    print(
                        f"Saved tokenized data chunk no. {chunks} to {tokens_path}/tokens_chunk_{chunks}.bin"
                    )
                    print(f"Len {cnt} tokens in this chunk, MBs:  {cnt * 4 / 1e6:.2f}")

                    cnt = 0
                    chunks += 1
                    texts = []

        else:
            all_train_texts = [doc["text"] for doc in dataset]

            encoded_inputs = tokenizer.encode(
                all_train_texts, add_bos=True, add_eos=True
            )

            longs_ids = list()
            for ids in encoded_inputs:
                longs_ids.append(np.array(ids["input_ids"], dtype=np.int32))
            print("Saving tokenized data")
            np.concatenate(longs_ids).tofile(
                os.path.join(tokens_path, f"{split}_tokens.bin")
            )
            print(f"Saved tokenized data to {tokens_path}/{split}_tokens.bin")
            del all_train_texts
            del longs_ids
            del encoded_inputs

    else:
        print("Tokenized data found, skipping creation")

    tokens_paths = []
    os_list = os.listdir(tokens_path)
    for file in os_list:
        if file.endswith(".bin"):
            tokens_paths.append(os.path.join(tokens_path, file))

    if len(tokens_paths) == 1:
        return tokens_paths[0]
    else:
        # merge chunks
        print(f"Merging {len(tokens_paths)} chunks...")
        merged_path = os.path.join(tokens_path, f"{split}_tokens.bin")

        # Sort to ensure correct order (tokens_0.bin, tokens_1.bin, ...)
        tokens_paths.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))

        with open(merged_path, "wb") as outfile:
            for chunk_path in tokens_paths:
                with open(chunk_path, "rb") as infile:
                    outfile.write(infile.read())
                os.remove(chunk_path)  # Clean up chunk files

        print(f"Merged to {merged_path}")
        return merged_path


def prepare_timeseries_tokenized_data(
    tokenizer: TSContinuousTokenizer | TSQuantizedTokenizer,
    tokens_path: str,
    hf_cache_path: str,
    path: str,
    name: str,
    target: str,
    regenerate: bool = False,
    streaming: bool = False,
    split: str = "train",
    chunk_toklimit: int = 250_000_000,  # 1GB per file
    fit_on_n_samples: int = 1_000_000,  # subsample to fit tokenizer
    **kwargs,
) -> str:
    """
    Will prepare tokenized data if not already present.
    Returns path to tokenized data.

    For now we merge all data into single binary file. For simplicity.
    """
    if not os.path.exists(tokens_path) or regenerate:
        if regenerate:
            if os.path.exists(tokens_path):
                print("Removing old data")
                shutil.rmtree(tokens_path)

        print("Tokenized data not found, creating...")
        os.makedirs(tokens_path)

        dataset = load_dataset(
            path, name=name, cache_dir=hf_cache_path, split=split, streaming=streaming
        )

        ts = []
        length = 0
        for x in dataset:
            val = x[target]
            length += 1
            ts.append(val)

        ts = np.array(ts, dtype=np.float32)
        print(f"Ts shape; {ts.shape}, total length: {length}")

        if tokenizer.requires_fit():
            tokenizer.fit(
                np.random.choice(ts, size=min(fit_on_n_samples, length), replace=False)
            )
        encoded_inputs = tokenizer.encode(ts)
        np_arr = np.array(
            [x["input_ids"] for x in encoded_inputs], dtype=tokenizer.dtype
        )
        print(f"Total tokens: {len(np_arr)}, size in MBs: {len(np_arr) * 4 / 1e6:.2f}")
        print(f"Shape {np_arr.shape}")
        np_arr.tofile(os.path.join(tokens_path, f"{split}_tokens.bin"))
        print(f"Saved tokenized data to {tokens_path}/{split}_tokens.bin")
    else:
        print("Tokenized data found, skipping creation")
    return os.path.join(tokens_path, f"{split}_tokens.bin")
