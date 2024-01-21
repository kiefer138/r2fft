# Standard library imports
import random
import timeit
from typing import Callable, List, Optional, Tuple

# Third party imports
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit  # type: ignore
from graphviz import Digraph  # type: ignore

# Local application imports
from r2fft.fft import FFTComputationTree, Node  # type: ignore
from r2fft.transform import dft, fft  # type: ignore


def plot_tree(node: Node, graph: Optional[Digraph] = None) -> Digraph:
    """
    Recursively traverse the tree and add each node and its edges to
    the graph
    """
    # If no graph is provided, create a new one
    if graph is None:
        graph = Digraph(name="FFT Computation Tree", format="png")

    # Add the current node to the graph
    graph.node(str(id(node)), label=str(node.data))

    # Add the left and right children to the graph
    if node.left is not None:
        graph.edge(str(id(node)), str(id(node.left)))
        plot_tree(node.left, graph)

    if node.right is not None:
        graph.edge(str(id(node)), str(id(node.right)))
        plot_tree(node.right, graph)

    return graph


def nlogn(n: np.ndarray, a: float) -> np.ndarray:
    return a * n * np.log(n)


def n2(n: np.ndarray, a: float) -> np.ndarray:
    return a * n**2


def performance_test_fft(max_power: int = 11) -> Tuple[List[int], List[float]]:
    # List of input sizes to test
    input_sizes = [2**i for i in range(1, max_power)]

    # List to store the elapsed times
    elapsed_times = []

    for size in input_sizes:
        # Generate a list of random complex numbers
        complex_numbers = [
            complex(random.random(), random.random()) for _ in range(size)
        ]

        # Create the FFT computation tree
        tree = FFTComputationTree(complex_numbers)

        # Measure the time it takes to compute the FFT
        elapsed_time = timeit.timeit(tree.compute, number=1)

        # Add the elapsed time to the list
        elapsed_times.append(elapsed_time)

    return input_sizes, elapsed_times


def performance_test_transform_func(
    transform_func: Callable, max_power: int = 11
) -> Tuple[List[int], List[float]]:
    # List of input sizes to test
    input_sizes = [2**i for i in range(1, max_power)]

    # List to store the elapsed times
    elapsed_times = []

    for size in input_sizes:
        # Generate a list of random complex numbers
        complex_numbers = [
            complex(random.random(), random.random()) for _ in range(size)
        ]

        # Measure the time it takes to execute the transform function
        start_time = timeit.default_timer()
        transform_func(complex_numbers)
        elapsed_time = timeit.default_timer() - start_time

        # Add the elapsed time to the list
        elapsed_times.append(elapsed_time)

    return input_sizes, elapsed_times


def performance_plot(filename, input_sizes, elapsed_times, fit_func):
    # Convert the lists to numpy arrays
    input_sizes_np = np.array(input_sizes)
    # Fit the function n*log(n) to the data
    params, params_covariance = curve_fit(fit_func, input_sizes_np, elapsed_times)

    # Calculate the standard deviation of the parameter
    param_std_dev = np.sqrt(np.diag(params_covariance))

    # Calculate the residuals and the fit error
    residuals = elapsed_times - fit_func(input_sizes_np, params[0])
    fit_error = np.sum(residuals**2)

    # Calculate the total sum of squares
    total_sum_of_squares = np.sum((elapsed_times - np.mean(elapsed_times)) ** 2)

    # Calculate R^2
    r_squared = 1 - (fit_error / total_sum_of_squares)

    # Plot the elapsed time versus input size
    plt.grid(True)
    plt.plot(input_sizes, fit_func(input_sizes_np, params[0]), "r--")
    plt.plot(input_sizes, elapsed_times, "ko")
    plt.xlabel("Input size")
    plt.ylabel("Elapsed time (s)")
    plt.title("FFT computation time")

    # Display the fit parameters, parameter error, fit error, and R^2 on the graph
    plt.text(
        0.05,
        0.95,
        f"a = {params[0]:.2e} ± {param_std_dev[0]:.2e}, R² = {r_squared:.2f}",
        transform=plt.gca().transAxes,
    )

    plt.savefig(filename)
    plt.close()


def random_periodic_data(size: int) -> Tuple[List[float], List[float]]:
    t = np.linspace(0, size, size)
    signal = np.zeros(size)

    for _ in range(1, np.random.randint(3, 8)):
        # Each sine wave has a frequency that is an integer multiple of the base frequency
        base_freq = 1 / size  # base frequency
        freq_mult = np.random.randint(1, 10)  # frequency multiplier
        freq = base_freq * freq_mult

        amplitude = np.random.uniform(0.1, 1)
        phase = np.random.uniform(0, 2 * np.pi)
        signal += amplitude * np.sin(2 * np.pi * freq * t + phase)

    return t.tolist(), signal.tolist()


def gaussian_peak(
    mu: float, sigma: float, num_points: int
) -> Tuple[List[float], List[float]]:
    x = np.linspace(mu - 3 * sigma, mu + 3 * sigma, num_points)
    y = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-((x - mu) ** 2) / (2 * sigma**2))
    return x.tolist(), y.tolist()


if __name__ == "__main__":
    # Define the input size
    size = 2**11

    # Generate the data
    time, data = random_periodic_data(size)

    # Create the plot
    fig, axs = plt.subplots(1, 2, figsize=(20, 6))

    # Adjust the spacing between subplots
    fig.subplots_adjust(left=0.05, right=0.95, wspace=0.2)

    # Plot the signal
    axs[0].plot(time, data, "k-")
    axs[0].set_title("Random Periodic Signal")
    axs[0].set_xlabel("Time (s)")
    axs[0].set_ylabel("Amplitude")
    axs[0].grid(True)

    # Compute the FFT of the data
    fft_tree = FFTComputationTree([complex(x) for x in data], extra_padding=0)

    # Compute the FFT of the data using fft_tree.compute()
    fft_data = np.array(fft_tree.compute())

    # Normalize the FFT data
    fft_data /= len(fft_data)

    # Compute the power spectrum
    power_spectrum = np.abs(fft_data) ** 2

    # Plot the power spectrum (only the first half for positive frequencies)
    axs[1].plot(power_spectrum[: len(power_spectrum) // 2], "k-")
    axs[1].set_title("Power Spectrum")
    axs[1].set_xlabel("Frequency (Hz)")
    axs[1].set_xlim(0, 120)
    axs[1].set_ylabel("Power")
    axs[1].grid(True)

    # Save the plot to a PNG file
    plt.savefig("signal_and_power_spectrum.png")
    plt.close()
