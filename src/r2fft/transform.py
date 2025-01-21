# Standard library imports
import cmath
from typing import List


def dft(signal: List[complex]) -> List[complex]:
    """
    Compute the Discrete Fourier Transform (DFT) of the input list.
    """
    # Get the length of the input signal
    samples: int = len(signal)

    # Compute the DFT of the input signal
    return [
        sum(
            signal[n] * cmath.exp(-2j * cmath.pi * k * n / samples)
            for n in range(samples)
        )
        for k in range(samples)
    ]


def fft(signal: List[complex]) -> List[complex]:
    """
    Compute the FFT of the input list using the Cooley-Tukey FFT algorithm.
    """
    samples: int = len(signal)  # The size of the input list

    # Base case: if the list is of length 1, return the list itself
    if samples == 1:
        return signal

    # Recursive case: divide by even and odd indices
    even: List[complex] = fft(signal[0::2])
    odd: List[complex] = fft(signal[1::2])

    # Multiply the odd-indexed elements with the phase factors
    c: List[complex] = [
        cmath.exp(-2j * cmath.pi * k / samples) * odd[k] for k in range(samples // 2)
    ]

    # Combine the even and modified odd-indexed elements
    return [even[k] + c[k] for k in range(samples // 2)] + [
        even[k] - c[k] for k in range(samples // 2)
    ]
