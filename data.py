import os

import numpy as np
from datasets import load_dataset

from hnet.utils.tokenizers import ByteTokenizer


def prepare_tokenized_data(
    tokenizer: ByteTokenizer,
    tokens_path: str,
    hf_cache_path: str,
    regenerate: bool = False,
    path: str = "Salesforce/wikitext",
    name: str = "wikitext-103-v1",
    streaming: bool = False,
    take_rows: int | None = None,
    chunk_toklimit: int = 250_000_000,  # 1GB per file
) -> str:
    """
    Will prepare tokenized data if not already present.
    Returns path to tokenized data.

    For now we merge all data into single binary file. For simplicity.
    """
    if not os.path.exists(tokens_path) or regenerate:
        print("Tokenized data not found, creating...")
        os.makedirs(tokens_path)

        dataset = load_dataset(
            path, name=name, streaming=streaming, split="train", cache_dir=hf_cache_path
        )

        if streaming:
            all_train_texts = dataset.take(take_rows).to_list()
            cnt = 0
            chunks = 0
            texts = []
            for doc in all_train_texts:
                texts.append(doc["text"])
                cnt += len(doc["text"])
                if cnt >= chunk_toklimit:
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
                    print(
                        f"Len {cnt} tokens in this chunk, MBs:  {cnt * 4 / 1_000_000:.2f}"
                    )

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
            np.concatenate(longs_ids).tofile(os.path.join(tokens_path, "tokens.bin"))
            print(f"Saved tokenized data to {tokens_path}/tokens.bin")

        del longs_ids
        del encoded_inputs
        del all_train_texts
        del dataset
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
        merged_path = os.path.join(tokens_path, "tokens.bin")

        # Sort to ensure correct order (tokens_0.bin, tokens_1.bin, ...)
        tokens_paths.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))

        with open(merged_path, "wb") as outfile:
            for chunk_path in tokens_paths:
                with open(chunk_path, "rb") as infile:
                    outfile.write(infile.read())
                os.remove(chunk_path)  # Clean up chunk files

        print(f"Merged to {merged_path}")
        return merged_path
