CHANGELOG
=========

PyPI pythonic-fp-numpy project.

Semantic Versioning
-------------------

Strict 3 digit semantic versioning

- **MAJOR** version incremented for incompatible API changes
- **MINOR** version incremented for backward compatible added functionality
- **PATCH** version incremented for backward compatible bug fixes

See `Semantic Versioning 2.0.0 <https://semver.org>`_.

Releases and Important Milestones
---------------------------------

PyPI v0.1.2 - 2026-01-16
~~~~~~~~~~~~~~~~~~~~~~~~

Fixed some documentation rough edges. Added *.pyi stub files. Updated __repr__
behavior for HWrapNDArray derived classes. Eliminated spaces.

Future directions.

- define ``+``, ``*``, and ``@`` on wrapped arrays.
- more tests


PyPI v0.1.1 - 2025-12-01
~~~~~~~~~~~~~~~~~~~~~~~~

Fixed problem with the extra test dependencies.


PyPI v0.1.0 - 2025-11-30
~~~~~~~~~~~~~~~~~~~~~~~~

Initial PyPI release.

Update - 2025-11-26
~~~~~~~~~~~~~~~~~~~

Created GitHub repo for this new effort. 

Previously wrote a hashable wrapper for NumPy NDArray for the test suite
of my boring-math-abstract-algebra PyPI project.

Makes an NDArray readonly and hashable.

Will need to drop the "only requires the Python std library" requirement
for pythonic-fp projects.
