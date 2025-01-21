# Standard library imports
import cmath
from math import ceil, log2
from typing import List, Optional


class Node:
    """
    A class to represent a node in the FFT computation tree
    """

    def __init__(self, data: List[complex]):
        # The data of the node
        self.data = data
        # The size of the data
        self.size = len(data)
        # The child nodes
        self.left: Optional[Node] = None
        self.right: Optional[Node] = None


class FFTComputationTree:
    """
    A class to represent the FFT computation tree
    """

    def __init__(self, signal: List[complex], extra_padding: Optional[int] = None):
        if extra_padding:
            signal.extend([0] * extra_padding)
        # Pad the signal to the next power of two
        samples = len(signal)
        next_power_of_two = 2 ** ceil(log2(samples))
        signal.extend([0] * (next_power_of_two - samples))

        # Create a Node from the signal and build the tree from it
        self.root: Node = self._build_tree(Node(signal))

    def _build_tree(self, node: Node) -> Node:
        """
        Build the tree structure based on the input list of complex numbers
        """
        # Get the size of the data
        samples: int = node.size

        # If the size is greater than 1, create left and right children
        if samples > 1:
            node.left = self._build_tree(Node(node.data[0::2]))
            node.right = self._build_tree(Node(node.data[1::2]))

        return node

    def compute(self, node: Optional[Node] = None) -> List[complex]:
        """
        Compute the FFT of the data in the tree
        """
        if node is None:
            node = self.root

        # If the size is 1, the FFT is the same as the data itself
        if node.size == 1:
            return node.data

        # If the size is greater than 1, recursively compute the FFT of
        # the left and right children
        even = self.compute(node.left)
        odd = self.compute(node.right)

        # Odd elements are multiplied by the phase factors
        c: List[complex] = [
            cmath.exp(-2j * cmath.pi * k / node.size) * odd[k]
            for k in range(node.size // 2)
        ]

        dft = [even[k] + c[k] for k in range(node.size // 2)] + [
            even[k] - c[k] for k in range(node.size // 2)
        ]
        return dft


if __name__ == "__main__":
    # Define some non-empty data
    data: List[complex] = [
        4 + 7j,
        9 + 9j,
        9 + 4j,
        3 + 0j,
        8 + 9j,
        6 + 7j,
        4 + 2j,
        9 + 9j,
    ]

    # Initialize a FFTComputationTree with the data
    tree: FFTComputationTree = FFTComputationTree(data)

    dft = tree.compute()
