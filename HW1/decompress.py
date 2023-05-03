import sys

import numpy as np
import pandas as pd
from tqdm import trange

pd.options.mode.chained_assignment = None

MIN_TIMESTAMP = 1670192818
UNCOMPRESSED = "data_to_compress.csv"
COMPRESSED = "compressed_data.dat"


def compress():
    df = pd.read_csv(UNCOMPRESSED).iloc[:10]
    df["uint16"] = 0
    df["uint32"] = 0
    for i in trange(df.shape[0]):
        uint24_1 = df["category_id"][i] + (2**11) * (
            df["microcategory_id"][i] + (2**11) * df["class"][i]
        )
        uint24_2 = df["timestamp"][i] - MIN_TIMESTAMP
        df["uint16"][i] = uint24_1 % (2**16)
        df["uint32"][i] = uint24_1 // (2**16) + 256 * uint24_2
        print(f"{uint24_1 // (2**16)=}, {uint24_2=}")
        int40 = int(str(df["item_id"][i]), 16)
        df["item_id"][i] = int40 // 256  # int32
        df["class"][i] = int40 % 256  # int8
    df["user_id"] = df["user_id"].astype("category")
    del df["category_id"]
    del df["microcategory_id"]
    df["class"] = df["class"].astype(np.uint8)
    df["uint16"] = df["uint16"].astype(np.uint16)
    df["uint32"] = df["uint32"].astype(np.uint32)
    df.to_pickle(COMPRESSED, compression="bz2")


def decompress():
    df = pd.read_pickle(COMPRESSED, compression="bz2")
    print(df)
    df["category_id"] = 0
    df["microcategory_id"] = 0
    for i in trange(df.shape[0]):
        int40 = df["class"][i] + 256 * df["item_id"][i]
        df["item_id"][i] = f"{int40:x}"
        uint24_1 = df["uint16"][i] + (2**16) * (df["uint32"][i] % 256)
        uint24_2 = df["uint32"][i] // 256
        df["timestamp"][i] = uint24_2 + MIN_TIMESTAMP
        df["category_id"][i] = uint24_1 % (2**11)
        uint24_1 //= (2**11)
        df["microcategory_id"][i] = uint24_1 % (2**11)
        df["class"][i] = uint24_1 // (2**11)
    df = df[
        [
            "user_id",
            "item_id",
            "category_id",
            "microcategory_id",
            "location_id",
            "timestamp",
            "model_a_score",
            "model_b_score",
            "price",
            "class",
        ]
    ]
    df.to_csv(UNCOMPRESSED, sep=",", index=False)


def main():
    if "decompress" in sys.argv[0].lower():
        print("Decompressing.", end="")
        decompress()
        print(" Done.")
    elif "compress" in sys.argv[0].lower():
        print("Compressing.", end="")
        compress()
        print(" Done.")
    else:
        print(
            "This program works the way you name it. It can:\n"
            "compress: data_to_compress.csv -> compressed_data.dat\n"
            "decompress: compressed_data.dat -> data_to_compress.csv"
        )


if __name__ == "__main__":
    main()
