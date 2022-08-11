# Dual Condensate Hamiltonian

## Getting Started
Create a Python Virtual Environment
```
python -m venv nameofyourvirtualenvironment

```
Enter the Virtual Environment

```
source nameofyourvirtualenvironment/bin/activate
```
Move into the directory (DC directory) and install the package with

```
python -m pip install .
```

If you think you are going to want to edit the source code, and not just use the package, type

```
python -m pip install -e .
```
instead.

## Usage
The code comes with two options for simulating the system.
- Densed
	- This method renders the Hamiltonian as full matrices, which can be much faster for smaller system sizes, but is quite memory intensive for larger systems.

- Sparsed
	- This method uses the [Sparse](https://sparse.pydata.org/en/stable/) tensor library to render the Hamiltonian in a Sparse Format. This is slower than the Dense method for smaller particle numbers.  

The [example file](example.py) contains a simple example for generating a convex hull of the 2-RDM from randomly sampling the Hamiltonian parameter space.


