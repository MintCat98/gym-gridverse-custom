from typing import Sequence, Type

import pytest

from gym_gridverse.envs.visibility_functions import (
    factory,
    fully_transparent,
    minigrid,
    partially_occluded,
    raytracing,
)
from gym_gridverse.geometry import Position
from gym_gridverse.grid import Grid
from gym_gridverse.grid_object import Floor, GridObject, Wall


@pytest.mark.parametrize(
    'objects',
    [
        [
            [Floor(), Floor(), Floor()],
            [Floor(), Floor(), Floor()],
            [Floor(), Floor(), Floor()],
        ],
        [
            [Wall(), Wall(), Wall()],
            [Wall(), Wall(), Wall()],
            [Wall(), Wall(), Wall()],
        ],
    ],
)
def test_full_visibility(objects: Sequence[Sequence[GridObject]]):
    grid = Grid.from_objects(objects)
    for position in grid.area.positions():
        visibility = fully_transparent(grid, position)

        assert visibility.dtype == bool
        assert visibility.all()


@pytest.mark.parametrize(
    'objects,position,expected_int',
    [
        (
            [
                [Floor(), Floor(), Floor(), Floor(), Floor()],
                [Floor(), Floor(), Floor(), Floor(), Floor()],
                [Floor(), Floor(), Floor(), Floor(), Floor()],
            ],
            Position(2, 2),
            [
                [1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1],
            ],
        ),
        (
            [
                [Floor(), Floor(), Floor(), Floor(), Floor()],
                [Floor(), Wall(), Wall(), Wall(), Floor()],
                [Floor(), Wall(), Floor(), Wall(), Floor()],
            ],
            Position(2, 2),
            [
                [0, 0, 0, 0, 0],
                [0, 1, 1, 1, 0],
                [0, 1, 1, 1, 0],
            ],
        ),
        (
            [
                [Floor(), Floor(), Floor(), Floor(), Floor()],
                [Floor(), Wall(), Floor(), Wall(), Floor()],
                [Floor(), Wall(), Floor(), Wall(), Floor()],
            ],
            Position(2, 2),
            [
                [1, 1, 1, 1, 1],
                [0, 1, 1, 1, 0],
                [0, 1, 1, 1, 0],
            ],
        ),
        (
            [
                [Floor(), Floor(), Floor(), Floor(), Floor()],
                [Floor(), Wall(), Wall(), Wall(), Floor()],
                [Floor(), Floor(), Floor(), Wall(), Floor()],
            ],
            Position(2, 2),
            [
                [1, 0, 0, 0, 0],
                [1, 1, 1, 1, 0],
                [1, 1, 1, 1, 0],
            ],
        ),
    ],
)
def test_partial_visibility(
    objects: Sequence[Sequence[GridObject]],
    position: Position,
    expected_int: Sequence[Sequence[int]],
):
    grid = Grid.from_objects(objects)
    visibility = partially_occluded(grid, position)
    assert visibility.dtype == bool
    assert (visibility == expected_int).all()


@pytest.mark.parametrize(
    'objects,position,expected_int',
    [
        (
            [
                [Floor(), Floor(), Floor(), Floor(), Floor()],
                [Floor(), Floor(), Floor(), Floor(), Floor()],
                [Floor(), Floor(), Floor(), Floor(), Floor()],
            ],
            Position(2, 2),
            [
                [1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1],
            ],
        ),
        (
            [
                [Floor(), Floor(), Floor(), Floor(), Floor()],
                [Floor(), Wall(), Wall(), Wall(), Floor()],
                [Floor(), Wall(), Floor(), Wall(), Floor()],
            ],
            Position(2, 2),
            [
                [0, 0, 0, 0, 0],
                [0, 1, 1, 1, 0],
                [0, 1, 1, 1, 0],
            ],
        ),
        (
            [
                [Floor(), Floor(), Floor(), Floor(), Floor()],
                [Floor(), Wall(), Floor(), Wall(), Floor()],
                [Floor(), Wall(), Floor(), Wall(), Floor()],
            ],
            Position(2, 2),
            [
                [1, 1, 1, 1, 1],
                [0, 1, 1, 1, 0],
                [0, 1, 1, 1, 0],
            ],
        ),
        (
            [
                [Floor(), Floor(), Floor(), Floor(), Floor()],
                [Floor(), Wall(), Wall(), Wall(), Floor()],
                [Floor(), Floor(), Floor(), Wall(), Floor()],
            ],
            Position(2, 2),
            [
                [1, 1, 1, 1, 1],
                [1, 1, 1, 1, 0],
                [1, 1, 1, 1, 0],
            ],
        ),
    ],
)
def test_minigrid_visibility(
    objects: Sequence[Sequence[GridObject]],
    position: Position,
    expected_int: Sequence[Sequence[int]],
):
    grid = Grid.from_objects(objects)
    visibility = minigrid(grid, position)
    assert visibility.dtype == bool
    assert (visibility == expected_int).all()


@pytest.mark.parametrize(
    'objects,position,expected_int',
    [
        (
            [
                [Floor(), Floor(), Floor(), Floor(), Floor()],
                [Floor(), Floor(), Floor(), Floor(), Floor()],
                [Floor(), Floor(), Floor(), Floor(), Floor()],
            ],
            Position(2, 2),
            [
                [1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1],
            ],
        ),
        (
            [
                [Floor(), Floor(), Floor(), Floor(), Floor()],
                [Floor(), Wall(), Wall(), Wall(), Floor()],
                [Floor(), Wall(), Floor(), Wall(), Floor()],
            ],
            Position(2, 2),
            [
                [0, 0, 0, 0, 0],
                [0, 1, 1, 1, 0],
                [0, 1, 1, 1, 0],
            ],
        ),
        (
            [
                [Floor(), Floor(), Floor(), Floor(), Floor()],
                [Floor(), Wall(), Floor(), Wall(), Floor()],
                [Floor(), Wall(), Floor(), Wall(), Floor()],
            ],
            Position(2, 2),
            [
                [0, 1, 1, 1, 0],
                [0, 1, 1, 1, 0],
                [0, 1, 1, 1, 0],
            ],
        ),
        (
            [
                [Floor(), Floor(), Floor(), Floor(), Floor()],
                [Floor(), Wall(), Wall(), Wall(), Floor()],
                [Floor(), Floor(), Floor(), Wall(), Floor()],
            ],
            Position(2, 2),
            [
                [0, 0, 0, 0, 0],
                [1, 1, 1, 1, 0],
                [1, 1, 1, 1, 0],
            ],
        ),
    ],
)
def test_raytracing_visibility(
    objects: Sequence[Sequence[GridObject]],
    position: Position,
    expected_int: Sequence[Sequence[int]],
):
    grid = Grid.from_objects(objects)
    visibility = raytracing(grid, position)
    assert visibility.dtype == bool
    assert (visibility == expected_int).all()


@pytest.mark.parametrize(
    'name',
    [
        'fully_transparent',
        'partially_occluded',
        'minigrid',
        'raytracing',
        'stochastic_raytracing',
    ],
)
def test_factory_valid(name: str):
    factory(name)


@pytest.mark.parametrize(
    'name,exception',
    [
        ('invalid', ValueError),
    ],
)
def test_factory_invalid(name: str, exception: Type[Exception]):
    with pytest.raises(exception):
        factory(name)
