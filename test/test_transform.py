# Standard library imports
from typing import List

# Related third party imports
import pytest
import numpy as np

# Local application/library specific imports
from r2fft.transform import dft, fft  # type: ignore


def test_dft() -> None:
    """
    Test the dft function with a simple list of complex numbers
    """
    # Define a simple list of complex numbers
    x: List[complex] = [complex(i) for i in range(8)]

    # Compute the DFT of the list using the dft function
    result: List[complex] = dft(x)

    # Compute the FFT of the list using numpy's fft function for comparison
    expected: np.ndarray = np.fft.fft(x)

    # Check if the result is close to the expected result
    assert np.allclose(result, expected)


def test_fft() -> None:
    """
    Test the fft function with a simple list of complex numbers
    """
    # Define a simple list of complex numbers
    x: List[complex] = [complex(i) for i in range(8)]

    # Compute the FFT of the list using the fft function
    result: List[complex] = fft(x)

    # Compute the FFT of the list using numpy's fft function for comparison
    expected: np.ndarray = np.fft.fft(x)

    # Check if the result is close to the expected result
    assert np.allclose(result, expected)


if __name__ == "__main__":
    pytest.main(["-v", __file__])
