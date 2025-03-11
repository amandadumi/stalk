#!/usr/bin/env python

__author__ = "Juha Tiihonen"
__email__ = "tiihonen@iki.fi"
__license__ = "BSD-3-Clause"


from pytest import raises


def disable_nexus():
    from sys import path
    # Temporarily remove nexus from path
    for p in path:
        if '/nexus/lib' in p:
            path.remove(p)
        # end if
    # end for
# end def


# Test Nexus being disabled
def test_nexus_disabled():
    disable_nexus()
    from stalk import nexus
    assert not nexus.nexus_enabled
    # Trying to import specific Nexus classes raises ModuleNotFoundError
    with raises(ModuleNotFoundError):
        from stalk.nexus import NexusPes
        NexusPes()
    # end with
# end def
