***************
Create the docs
***************

pydoc
=====

I wonder if you really need to create a source documentaion --
pydoc is a wonderful tool.

You can list all package members::

   python -m pydoc reduction

or show a special one::

   python -m pydoc reduction.normalize


sphinx
======

Okay, if you really need to...

I found sphinx-quickstart not very helpful to create an initial project.
You can use sphinx-apidoc to do all in one place::

   sphinx-apidoc -Faf -o source ../ -A $(git config user.name) --extension sphinx.ext.napoleon

This will

* parse all packages in `..` recursively
* creates a now sphinx documentaion project in `source` and
* uses `git user.name` as author.

After this you can e.g. create and display the `html` documentation::

   make -C source html
   browse source/_build/html/index.html


Useful links
============

* https://docs.python.org/3/library/pydoc.html
* https://www.sphinx-doc.org/en/master/index.html

Cheers
--ChristianWBrock
