import numpy as np


class ByteTokenizer:
    def __init__(self):
        self.vocab_size = 256
        self.bos_idx = 254
        self.eos_idx = 255
        self.dtype = np.uint8

    def encode(
        self, seqs: list[str], add_bos: bool = False, add_eos: bool = False, **kwargs
    ) -> list[dict[str, np.ndarray]]:
        total_outputs = []
        for text in seqs:
            text_byte = text.encode("utf-8")

            if add_bos:
                text_byte = bytes([self.bos_idx]) + text_byte
            if add_eos:
                text_byte = text_byte + bytes([self.eos_idx])
            text_byte = bytearray(text_byte)
            text_byte_ids = np.array(text_byte, dtype=self.dtype)

            total_outputs.append({"input_ids": text_byte_ids})

        return total_outputs

    def decode(self, tokens: np.ndarray | list[int], **kwargs) -> str:
        if isinstance(tokens, np.ndarray):
            tokens = tokens.tolist()
        return bytearray(tokens).decode("utf-8", **kwargs)


class TSTokenizer:
    def requires_fit(self) -> bool:
        return False


class TSContinuousTokenizer(TSTokenizer):
    """
    This tokenizer is for continuous time series data.

    In practice, it just passes float32 arrays through so
    that the data pipeline remains consistent.
    """

    def __init__(self):
        super().__init__()
        self.dtype = np.float32

    def encode(self, seqs: list[np.ndarray], **kwargs) -> list[dict[str, np.ndarray]]:
        total_outputs = []
        for series in seqs:
            series_array = np.array(series, dtype=self.dtype)
            total_outputs.append({"input_ids": series_array})
        return total_outputs

    def decode(self, tokens: np.ndarray | list[float], **kwargs) -> np.ndarray:
        if isinstance(tokens, list):
            tokens = np.array(tokens, dtype=self.dtype)
        return tokens


class TSQuantizedTokenizer(TSTokenizer):
    """
    This tokenizer is for quantized time series data.

    It converts float32 arrays to uint8 arrays and vice versa.
    """

    def __init__(self, vocab_size: int = 256, quant_type: str = "equiwidth"):
        """
        quant_type can be 'equiwidth' or 'cdf'
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.dtype = np.uint8
        self.quant_type = quant_type
        self.fit_called = False

    def requires_fit(self):
        return True

    def fit(self, data: np.ndarray):
        """
        Fit the tokenizer on the data to determine quantization parameters.
        """
        self.fit_called = True
        self.data_min = np.min(data)
        self.data_max = np.max(data)
        if self.quant_type == "equiwidth":
            self.bin_edges = np.linspace(
                self.data_min, self.data_max, self.vocab_size + 1
            )
        elif self.quant_type == "cdf":
            self.bin_edges = np.percentile(
                data, np.linspace(0, 100, self.vocab_size + 1)
            )
        else:
            raise ValueError("Unsupported quantization type")

    def encode(self, seqs: list[np.ndarray], **kwargs) -> list[dict[str, np.ndarray]]:
        assert self.fit_called, "Tokenizer must be fitted before encoding. Call fit() with representative data."
        total_outputs = []
        for series in seqs:
            series_array = np.array(series, dtype=np.float32)
            quantized = np.digitize(series_array, self.bin_edges) - 1
            quantized = np.clip(quantized, 0, self.vocab_size - 1).astype(self.dtype)
            total_outputs.append({"input_ids": quantized})
        return total_outputs

    def decode(self, tokens: np.ndarray | list[int], **kwargs) -> np.ndarray:
        if isinstance(tokens, list):
            tokens = np.array(tokens, dtype=self.dtype)
        dequantized = (self.bin_edges[tokens] + self.bin_edges[tokens + 1]) / 2
        return dequantized.astype(np.float32)


if __name__ == "__main__":
    # Example usage and test
    print("Byte Tokenizer Test:")
    byte_tokenizer = ByteTokenizer()
    text = "Hello World"
    encoded = byte_tokenizer.encode([text])
    print("Encoded Byte:", encoded)
    decoded = byte_tokenizer.decode(encoded[0]["input_ids"])
    print("Decoded Byte:", decoded)

    print("\n ------- \n")
    print("Time Series Continuous Tokenizer Test:")

    ts_tokenizer = TSContinuousTokenizer()
    series = np.array([0.1, 0.5, 0.9], dtype=np.float32)
    encoded_ts = ts_tokenizer.encode([series])
    print("Encoded TS:", encoded_ts)
    decoded_ts = ts_tokenizer.decode(encoded_ts[0]["input_ids"])
    print("Decoded TS:", decoded_ts)

    print("\n ------- \n")
    print("Time Series Quantized Tokenizer (Equiwidth) Test:")
    series = np.array([1.5, 2.5, 4.0, 6.0, 7.5], dtype=np.float32)
    quant_tokenizer = TSQuantizedTokenizer(vocab_size=4, quant_type="equiwidth")
    quant_tokenizer.fit(np.array([0.0, 1.0, 2.0, 3.0, 4.0], dtype=np.float32))
    encoded_quant = quant_tokenizer.encode([series])
    print("Encoded Quantized TS:", encoded_quant)
    decoded_quant = quant_tokenizer.decode(encoded_quant[0]["input_ids"])
    print("Decoded Quantized TS:", decoded_quant)

    print("\n ------- \n")
    print("Time Series Quantized Tokenizer (CDF) Test:")
    quant_tokenizer_cdf = TSQuantizedTokenizer(vocab_size=4, quant_type="cdf")
    quant_tokenizer_cdf.fit(np.array([0.0, 1.0, 2.0, 3.0, 4.0], dtype=np.float32))
    encoded_quant_cdf = quant_tokenizer_cdf.encode([series])
    print("Encoded Quantized TS (CDF):", encoded_quant_cdf)
    decoded_quant_cdf = quant_tokenizer_cdf.decode(encoded_quant_cdf[0]["input_ids"])
    print("Decoded Quantized TS (CDF):", decoded_quant_cdf)
