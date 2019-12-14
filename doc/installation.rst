************
Installation
************

Install Python
==============

First we need to install python itself, as well as
[pip], **the** PYthon Package Index [PyPi] installer and the
[venv] module, required to create virtual environments.

For plotting reasons tk is also required.


On linux (debian or ubuntu)
---------------------------

Make sure to update the package list::

    sudo apt-get update

Now we install python, pip, venv and tk.::

    sudo apt-get install python3 python3-pip python3-venv python3-tk


On Windows
----------

Sorry, I recommend you figure how to do the same using [conda].
It has messy syntax, but works as well.

Create a new virtual environment
================================

Python is fast paced with some times less then optimal consideration for backward compatibility.
This means: the new library version you really need for project A may break project B.

The solution are *virtual environments*. I **strongly** recommend to use a **new** python environment for **each** project
you are working with. How to do this? Easy!

In the following I make some recommendations that worked for me.

 * All your venv are located in a $HOME/venv/ (or ~/venv/ for short).
 * venv should have meaningful names
 * the one I use below will be called 'spectro'.

Lets create the 'spectro' venv::

    python3 -m venv ~/venv/spectro

Now we should have:

* a folder named *~/venv/spectro* containing a separate python environment.
* *~/venv/spectro/bin/* folder contains the installed executables,
* *~/venv/spectro/lib/pythonX/site-packages/* the installed modules,
* etc.

Activate the new environment::

    . ~/venv/spectro/bin/activate

The leading **.** is important!

This command changes the system in some ways:

* The command prompt should have changed, showing the venv name
* Now all python commands will be executed in the *~/venv/spectro* environment
* Ask your shell which python is used::

    type python

Install the module
==================

To install the module from PyPi::

    pip install algol-reduction

If a friend or colleague sent you a new the wheel file you can also install it locally::

    pip install algol_reduction-1.0.0a10-py3-none-any.whl

You should see that not only the new package *reduction* is installed but also
all scripts and all dependent packages.
You can verify this::

    fitsheader --help             # installed by astropy
    fits_display1d --help         # a reduction script
    plan_observations --help      # the same


Have fun
--Christian Brock


Some useful links:

* https://conda.io/
* https://pypi.org/project/pip/
* https://pypi.org/
* https://www.python.org
* https://docs.python.org/3.5/library/venv.html
* https://www.python.org/dev/peps/pep-0427/

