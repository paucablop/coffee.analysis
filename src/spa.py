from io import BytesIO

import numpy as np
import warnings


SAMPLE_FLAG = 3
SAMPLE_INTERFEROGRAM_FLAG = 102
BACKGROUND_INTERFEROGRAM_FLAG = 103


def _data_from_bytebuffer(byte_array: BytesIO, flag_value: int):
    position = 0x000130
    size_address = 0x000126

    flag = 1
    byte_array.seek(size_address)
    size = np.frombuffer(byte_array.read(2), dtype=np.int16)[0]
    for i in range(0, size):
        byte_array.seek(position)
        flag = np.frombuffer(byte_array.read(2), dtype=np.int16)[0]
        if flag == flag_value:
            start = np.frombuffer(byte_array.read(4), dtype=np.int32)[0]
            length = np.frombuffer(byte_array.read(4), dtype=np.int32)[0]
            byte_array.seek(start)
            data = byte_array.read(length)
            float_data = np.frombuffer(data, dtype=np.float32)
            return float_data
        position += 16
    warnings.warn("Data array not found")
    return None


### FROM BYTE ARRAY ###
def read_absorbance_from_bytearray(byte_array) -> np.ndarray:
    absorbance: np.ndarray = _data_from_bytebuffer(BytesIO(byte_array), SAMPLE_FLAG)
    return np.flipud(absorbance)


def read_sample_interferogram_from_bytearray(byte_array: bytearray) -> np.ndarray:
    interferogram: np.ndarray = _data_from_bytebuffer(
        BytesIO(byte_array), SAMPLE_INTERFEROGRAM_FLAG
    )
    return interferogram


def read_background_interferogram_from_bytearray(byte_array: bytearray) -> np.ndarray:
    interferogram: np.ndarray = _data_from_bytebuffer(
        BytesIO(byte_array), BACKGROUND_INTERFEROGRAM_FLAG
    )
    return interferogram


### FROM SPA FILE ###
def read_absorbance(file_path: str) -> np.ndarray:
    # the floats in a SPA file are stored from 4000 to 700 wavenumbers
    # Reversed using np.flipud to read from 700 to 4000
    # like how wavenumbers are read in the CSV
    with open(file_path, "rb") as file:
        absorbance = _data_from_bytebuffer(file, SAMPLE_FLAG)
    return np.flip(absorbance)


def read_sample_interferogram(file_path: str) -> np.ndarray:
    with open(file_path, "rb") as file:
        interferogram = _data_from_bytebuffer(file, SAMPLE_INTERFEROGRAM_FLAG)
    return interferogram


def read_background_interferogram(file_path: str) -> np.ndarray:
    with open(file_path, "rb") as file:
        interferogram = _data_from_bytebuffer(file, BACKGROUND_INTERFEROGRAM_FLAG)
    return interferogram
