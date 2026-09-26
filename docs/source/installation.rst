Installation
============

Requirements
------------

* Python 3.12 or later
* PyTorch 2.12 or later

Directly from GitHub
--------------------

Install the package directly using ``pip`` without cloning the repository:

.. code-block:: bash

   pip install git+https://github.com/intensivedatacomp/artificial-dataset.git

To install with development dependencies:

.. code-block:: bash

   pip install "artificial-dataset[dev] @ git+https://github.com/intensivedatacomp/artificial-dataset.git"

To pin to a specific commit hash:

.. code-block:: bash

   pip install git+https://github.com/intensivedatacomp/artificial-dataset.git@<commit-hash>

To pin to a specific commit hash with development dependencies:

.. code-block:: bash

   pip install "artificial-dataset[dev] @ git+https://github.com/intensivedatacomp/artificial-dataset.git@<commit-hash>"

Using requirements.txt
----------------------

To add the package to ``requirements.txt`` with a fixed commit hash and development dependencies:

.. code-block:: text

   artificial-dataset[dev] @ git+https://github.com/intensivedatacomp/artificial-dataset.git@<commit-hash>

Using environment.yml (Conda)
-----------------------------

To include the package in a Conda ``environment.yml`` file, add it under the ``pip`` section:

.. code-block:: yaml

   name: my-env
   channels:
     - conda-forge
     - defaults
   dependencies:
     - python>=3.12
     - pip
     - pip:
         - "artificial-dataset[dev] @ git+https://github.com/intensivedatacomp/artificial-dataset.git@<commit-hash>"

From source
-----------

Clone the repository and install in editable mode:

.. code-block:: bash

   git clone https://github.com/intensivedatacomp/artificial-dataset.git
   cd artificial-dataset
   pip install -e .

To install optional development or documentation dependencies:

.. code-block:: bash

   pip install -e ".[dev]"
   # or
   pip install -e ".[docs]"
