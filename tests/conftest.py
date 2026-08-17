"""Test environment isolation.

Host-level git configuration (commit signing hooks, templates, ...) must
not leak into the temporary repositories the tests create; a signing
helper that needs a network service would make every checkpoint commit
flaky. Point git at empty global/system configs for the whole test run.
"""

import os

os.environ["GIT_CONFIG_GLOBAL"] = os.devnull
os.environ["GIT_CONFIG_SYSTEM"] = os.devnull
