Contributing
============

Ocean's `contributing guide <https://docs.dwavequantum.com/en/latest/ocean/contribute.html>`_
has guidelines for contributing to Ocean packages.

If you're interested in adding or modifying parameters of the ``find_embedding``
primary utility function, please see the `<parameter_checklist.txt>`_ file.

Release Notes
-------------

``minorminer`` makes use of `reno <https://docs.openstack.org/reno/>`_
to manage its release notes.

When making a contribution to ``minorminer`` that will affect users,
create a new release note file by running

.. code-block:: bash

    reno new your-short-descriptor-here

You can then edit the file created under ``releasenotes/notes/``. Remove any sections
not relevant to your changes. Commit the file along with your changes.

See reno's `user guide <https://docs.openstack.org/reno/latest/user/usage.html>`_
for details.
