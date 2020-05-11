import numpy as np

floating_dtypes = [np.float16, np.float32, np.float64]

integer_dtypes = [np.int8, np.int16, np.int32, np.int64]


def convert_dtype(array, dtype):
    dtype = np.dtype(dtype)
    
    if array.dtype == dtype:
        return array
    
    if dtype in integer_dtypes:
        item_size = dtype.itemsize
        if array.dtype in integer_dtypes:  # int -> int
            array_item_size = array.dtype.itemsize
            if array_item_size > item_size:
                array /= int(2 ** ((array_item_size - item_size) * 8))
                array = np.asarray(array, dtype)
            elif array_item_size < item_size:
                array = np.asarray(array, dtype)
                array *= int(2 ** ((item_size - array_item_size) * 8))
        elif array.dtype in floating_dtypes:  # float -> int
            array *= 2 ** (item_size * 8 - 1)
            array = np.asarray(array, dtype)
        else:
            raise TypeError("Unsupported array with dtype " + array.dtype.name)
    elif dtype in floating_dtypes:
        if array.dtype in integer_dtypes:  # int -> float
            array_item_size = array.dtype.itemsize
            array /= int(2 ** (array_item_size * 8))
            array = np.asarray(array, dtype)
        elif array.dtype in floating_dtypes:  # float -> float
            array = np.asarray(array, dtype)
        else:
            raise TypeError("Unsupported array with dtype " + array.dtype.name)
    else:
        raise TypeError("Unsupported dtype " + dtype.name)
    
    return array
