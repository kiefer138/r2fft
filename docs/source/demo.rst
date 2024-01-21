Radix 2 FFT algorithm
=====================

In this project, I will implement and analyze a specific implementation of fast Fourier transform (FFT), known as the Cooley-Tukey FFT algorithm for input sizes of :math:`2^m` data where :math:`m\in\mathbb{N}`. 
This algorithm is used to find the Fourier coefficients of a one-dimensional complex array with a size that is of power 2 and was published in 1965 by James Cooley from IBM and John Tukey of Princeton. 
The algorithm is used to find the so-called Fourier coefficients of the input signal which represent the amplitudes of frequency components in the original data as illustrated below (signal on left, Fourier coefficients on right).

Periodic time-series are representative of changes in the position of a sensor as in seismology, the electric field of a light signal as in optics, or even pressure waves as in audio applications. 
In the case of optics, the power spectrum tells us the component colors within the light wave, or in audio application, it tells us the component tones. 
There is also a biological connection to this type of signal analysis as our ears measure pressure waves and send these signals to a part of our brain called the auditory cortex where the signals are processed to perceive sounds and tones. 
Moreover, at the most fundamental level we have quantum mechanics which describes the position and momentum of a particle as Fourier transforms of one another which gives rise to the quantum uncertainty principle. 
The motivation for Cooley and Tukey to develop the algorithm was national defense as it was used to interpret seismological time-series data during the cold war and detect frequency components indicative of secretive nuclear testing, a motivation credited to Tukey. Now the algorithm is ubiquitous to time-series and signal analysis and used in audio processing, image analysis, communications, and various scientific and engineering fields.

The algorithm that we will be exploring in this project uses divide and conquer to speed the calculation up from :math:`O(n^2)` to :math:`O(n \log n)`, specifically it recursively divides the input list and calculates the discrete Fourier transforms on smaller arrays with :math:`O(n \log n)` performance beginning with the leaves and bubbling up towards the root. 

Fourier Series and the Discrete Fourier Transform
-------------------------------------------------

A discrete Fourier transform (DFT) sounds like it may just be a discretized version of a Fourier transform. This is incorrect because we are not calculating a Fourier transform but rather a Fourier series. The main difference is that a Fourier series requires the function to be periodic whereas a Fourier transform can be performed on non-periodic functions of infinite period.
In order to find the coefficients of the Fourier series of a given function we must integrate the function times the relevant frequency component. Since all other frequencies are orthogonal, this plucks out the relevant frequency in the series and the amplitude is the overlap of that frequency component with the function. It is this integral that becomes discretized when performing a DFT.
Any function satisfying the Dirichlet conditions can be expanded in a Fourier series.
Any function satisfying the Dirichlet conditions can be represented as a Fourier series; periodic, piecewise continuous, bounded, finite discontinuities, finite number of maxima and minima, and integrable.

The Fourier series of a function :math:`f(t)` that satisfies these conditions is expanded below.


.. math::

   f(t) = \frac{a_0}{2} + \sum_{n = 1}^{\infty} \left[ a_n \cos\left(\frac{2\pi n t}{T} \right) + b_n \sin\left(\frac{2\pi n t}{T} \right) \right]


All that's left to do is find the :math:`a_n` and :math:`b_n` required to express :math:`f(t)` in terms of :math:`\sin` and :math:`\cos`.


.. math::

   a_n = \frac{2}{T} \int^{t_0 + T}_{t_0} f(t) \cos\left(\frac{2\pi n t}{T} \right) \, dt

.. math::

   b_n = \frac{2}{T} \int^{t_0 + T}_{t_0} f(t) \sin\left(\frac{2\pi n t}{T} \right) \, dt


This can be simplified by rewriting in complex notation from Euler's formula, :math:`e^{i\theta} = \cos\theta + i \sin\theta`, and finding the new complex coefficients :math:`c_k = (a_k - ib_k)/2` and :math:`c_{-k} = (a_k + ib_k)/2`.


.. math::

   f(t) = \sum_{k=-\infty}^{\infty} c_k e^{i\omega_k t} \,\,\, \mathrm{where} \,\,\, \omega_k = \frac{2\pi }{T} k = \omega_0 k \,\,\,:\,\,\, k\in\mathbb{Z}

.. math::
   
   c_k = \frac{1}{T} \int_{t_0}^{t_0 + T} f(t) e^{-i\omega_k t} \, dt


Now we consider this integral for a discrete function by considering the Riemann sum. 
The time data is discretized :math:`t_n = t_0 + nT/N` where :math:`N` is the number of sample points within :math:`T`.

.. math::

   c_k = \frac{1}{T} \sum_{n=0}^{N-1} f(t_n) e^{-i\omega_k t_n} \left(\frac{T}{N}\right)

.. math::

   c_k = \frac{e^{-i\omega_0 t_0 k}}{N}  \sum_{n=0}^{N-1} f_n  e^{-i\omega_0 nk T/ N}


Since :math:`t_0` only affects the result by a phase factor we can set it to :math:`0` for simplicity.


.. math::
   
   c_k = \frac{1}{N} \sum_{n=0}^{N-1} f_n  e^{-2\pi i nk / N}


It is common practice to drop the :math:`1/N` factor during the DFT and reintroduce it during the inverse DFT. 
A matrix formulation of the above is given below.


.. math::

   \begin{pmatrix} c_0 \\ c_1 \\ c_2 \\ \vdots \\c_{N-1} \end{pmatrix} =
   \begin{pmatrix}
   1 & 1 & 1 & \cdots & 1 \\
   1 & e^{-2\pi i/N} & e^{-4\pi i/N} & \cdots & e^{-2\pi i (N-1)/N} \\
   1 & e^{-4\pi i/N} & e^{-8\pi i/N} & \cdots & e^{-4\pi i (N-1)/N} \\ 
   \vdots & \vdots & \vdots & \ddots & \vdots \\
   1 & e^{-2\pi i (N-1)/N} & e^{-4\pi i (N-1)/N} & \cdots & e^{-2\pi i (N-1)^2/N} \\ 
   \end{pmatrix}
   \begin{pmatrix} f_0 \\ f_1 \\ f_2 \\ \vdots \\ f_{N-1} \end{pmatrix}


Implementation of DFT
---------------------

.. math::

   c_k =  \sum_{n=0}^{N-1} f_n  e^{-2\pi i nk / N}

We must perform :math:`n` multiplications and sums :math:`n` times for an overall :math:`O(n^2)` for matrix multiplication.

You can include Python code blocks in your documentation. For example:


.. ipython:: python
   :suppress:

   from r2fft.fft import FFTComputationTree
   from r2fft.visualize import plot_tree

.. ipython:: python

   # Define a list of complex numbers
   data = [
       4 + 7j,
       9 + 9j,
       9 + 4j,
       3 + 0j,
       8 + 9j,
       6 + 7j,
       4 + 2j,
       9 + 9j,
   ]

.. ipython:: python
   :suppress:

   # Initialize a FFTComputationTree with the data
   tree = FFTComputationTree(data)

   # Plot the tree and save it to an image file
   graph = plot_tree(tree.root)
   graph.render(filename='fft_computation_tree', directory='source/savefig', cleanup=True)

Then, you can include the image in your documentation with the `image` directive:

.. image:: savefig/fft_computation_tree.png
   :width: 800
