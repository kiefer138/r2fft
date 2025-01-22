# Standard library imports
from typing import List

# Related third party imports
import pytest
import numpy as np

# Local application/library specific imports
from r2fft.fft import Node, FFTComputationTree  # type: ignore


def test_node_initialization():
    """
    This test checks that a Node is correctly initialized when it is
    given non-empty data
    """

    # Define some non-empty data
    data: List[complex] = [1 + 1j, 2 + 2j, 3 + 3j]
    # Initialize a Node with the data
    node: Node = Node(data)

    assert node.data == data
    assert node.size == len(data)
    assert node.left is None
    assert node.right is None


def test_node_initialization_empty():
    """
    This test checks that a Node is correctly initialized when it is
    given empty data
    """

    # Define some empty data
    data: List[complex] = []

    # Initialize a Node with the data
    node: Node = Node(data)

    assert node.data == data
    assert node.size == 0

    assert node.left is None
    assert node.right is None


def test_fft_computation_tree_initialization():
    """
    This test checks that a FFTComputationTree is correctly initialized
    when it is given non-empty data
    """

    # Define some non-empty data
    data: List[complex] = [1 + 1j, 2 + 2j, 3 + 3j]

    # Initialize a FFTComputationTree with the data
    tree: FFTComputationTree = FFTComputationTree(data)

    assert tree.root.data == data
    assert tree.root.size == len(data)


def test_fft_computation_tree_compute():
    """
    This test checks that the compute method correctly computes the FFT
    of the input data
    """

    # Define some non-empty data
    data: List[complex] = [1 + 1j, 2 + 2j, 3 + 3j, 4 + 4j]

    # Initialize a FFTComputationTree with the data
    tree: FFTComputationTree = FFTComputationTree(data)

    # Compute the FFT of the data
    computed_fft: List[complex] = tree.compute()

    # The expected FFT of the data
    expected_fft: List[complex] = np.fft.fft(data)

    assert computed_fft == pytest.approx(expected_fft, abs=1e-6)


if __name__ == "__main__":
    pytest.main(["-v", __file__])
