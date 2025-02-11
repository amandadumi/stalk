#!/usr/bin/env python3
'''Class for containing a 1D grid of points, values and errorbars
'''

from numpy import array

from stalk.params.LineSearchPoint import LineSearchPoint

__author__ = "Juha Tiihonen"
__email__ = "tiihonen@iki.fi"
__license__ = "BSD-3-Clause"


class LineSearchGrid():
    # List of LineSearchPoint instances
    _grid: list[LineSearchPoint] = []

    def __init__(
        self,
        grid=None
    ):
        if grid is not None:
            for offset in grid:
                self.add_point(offset)
            # end for
        # end for
    # end def

    def add_point(self, point):
        if not isinstance(point, LineSearchPoint):
            point = LineSearchPoint(point)
        # end if
        if self.find_point(point) is None:
            self._grid.append(point)
            # Keep the grid sorted
            self._grid.sort()
        # end if
    # end def

    @property
    def valid_grid(self):
        '''Return offset array of valid points'''
        return array([point.offset for point in self._grid if point.valid])
    # end def

    @property
    def grid(self):
        '''Return offset array of points'''
        return array([point.offset for point in self._grid])
    # end def

    @grid.setter
    def grid(self, grid):
        self._grid = []
        for offset in grid:
            self.add_point(offset)
        # end for
    # end def

    @property
    def valid_values(self):
        '''Return values array of valid points'''
        return array([point.value for point in self._grid if point.valid])
    # end def

    @property
    def values(self):
        '''Return values array of points'''
        return array([point.value for point in self._grid])
    # end def

    @values.setter
    def values(self, values):
        if len(values) == len(self):
            for value, point in zip(values, self._grid):
                point.value = value
            # end for
        # end if
    # end def

    @property
    def valid_errors(self):
        '''Return errors array of valid points'''
        return array([point.error for point in self._grid if point.valid])
    # end def

    @property
    def errors(self):
        '''Return errors array of valid points'''
        return array([point.error for point in self._grid])
    # end def

    @errors.setter
    def errors(self, errors):
        if len(errors) == len(self):
            for error, point in zip(errors, self._grid):
                point.error = error
            # end for
        # end if
    # end def

    # Get all enabled arrays: (grid, values, errors)
    def get_all(self):
        return self.grid, self.values, self.errors
    # end def

    # Get full arrays including disabled and invalid: (grid, values, errors)
    def get_valid(self):
        return self.valid_grid, self.valid_values, self.valid_errors
    # end def

    # Finds and returns a requested point, if present; if not, returns None
    def find_point(self, point):
        # Try to find point by index
        if isinstance(point, int):
            if abs(point) < len(self):
                point = self._grid[point]
            else:
                return None
            # end if
        elif not isinstance(point, LineSearchPoint):
            # point is assumed to be a scalar offset
            point = LineSearchPoint(point)
        # end if
        # Else: point must be a LineSearchPoint
        for point_this in self._grid:
            if point_this == point:
                return point_this
            # end if
        # end for
    # end def

    # Sets the value and error for a given point, if present
    def set_value_error(self, offset, value, error=0.0):
        point = self.find_point(offset)
        if point is not None:
            point.value = value
            point.error = error
        # end if
    # end def

    # Enable a point by offset, if present
    def enable_value(self, offset):
        point = self.find_point(offset)
        if point is not None:
            point.enabled = True
        # end if
    # end def

    # Disable a point by offset, if present
    def disable_value(self, offset):
        point = self.find_point(offset)
        if point is not None:
            point.enabled = False
        # end if
    # end def

    def __len__(self):
        return len(self.grid)
    # end def

    # str of grid
    def __str__(self):
        if self.grid is None:
            string = '\n  data: no grid'
        else:
            string = '\n  data:'
            string += '\n    {:9s} {:9s} {:9s}'.format('grid', 'value', 'error')
            for point in zip(self.grid):
                string += str(point)
            # end for
        # end if
        return string
    # end def

# end class
