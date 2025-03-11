from pytest import raises
from stalk.io.GeometryLoader import GeometryLoader


def test_GeometryLoader():

    with raises(TypeError):
        GeometryLoader()
    # end with

    # Nominal functionality is tested in derived classes
# end def
