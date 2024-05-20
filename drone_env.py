from __future__ import annotations
from typing import Optional
import enum
import numpy.random as rnd

from gym_gridverse.action import Action
from gym_gridverse.agent import Agent
from gym_gridverse.design import draw_room, draw_wall_boundary
from gym_gridverse.envs.reset_functions import reset_function_registry
from gym_gridverse.envs.reward_functions import reward_function_registry
from gym_gridverse.envs.terminating_functions import (
    terminating_function_registry,
)
from gym_gridverse.envs.transition_functions import transition_function_registry
from gym_gridverse.geometry import Area, Orientation
from gym_gridverse.grid import Grid
from gym_gridverse.grid_object import Color, Floor, GridObject, Wall
from gym_gridverse.rng import choice, get_gv_rng_if_none
from gym_gridverse.state import State


class DeliverPoint(GridObject):
    state_index = 0
    color = Color.NONE
    blocks_movement = False
    blocks_vision = False
    holdable = True

    @classmethod
    def can_be_represented_in_state(cls) -> bool:
        return True

    @classmethod
    def num_states(cls) -> int:
        return 1

    def __repr__(self):
        return f'{self.__class__.__name__}()'

class Hub(GridObject):
    color = Color.YELLOW
    blocks_movement = False
    blocks_vision = False
    holdable = False
    item_num = 5

    class Status(enum.Enum):
        OPEN = 0
        CLOSED = 1

    def __init__(self):
        self.state = self.Status.OPEN

    @classmethod
    def can_be_represented_in_state(cls) -> bool:
        return True

    @classmethod
    def num_states(cls) -> int:
        return len(Hub.Status)
    
    @property
    def state_index(self) -> int:
        return self.state.value
        
    @property
    def is_open(self) -> bool:
        return self.state is Hub.Status.OPEN

    @property
    def is_closed(self) -> bool:
        return self.state is Hub.Status.CLOSED
    
    def set_closed(self):
        self.state = self.Status.CLOSED

    def __repr__(self):
        return f'{self.__class__.__name__}({self.state!s})'
    
@reset_function_registry.register
def coin_maze(*, rng: Optional[rnd.Generator] = None) -> State:
    """creates a maze with collectible coins"""

    # must call this to include reproduceable stochasticity
    rng = get_gv_rng_if_none(rng)

    # initializes grid with Coin
    grid = Grid.from_shape((7, 9), factory=DeliverPoint)
    # assigns Wall to the border
    draw_wall_boundary(grid)
    # draw other walls
    draw_room(grid, Area((2, 4), (2, 6)), Hub)
    # re-assign openings
    #grid[2, 3] = Floor()
    #grid[4, 5] = Coin()

    # final result (#=Wall, .=Coin):

    # #########
    # #.......#
    # #.W.WWW.#
    # #.W...W.#
    # #.WWW.W.#
    # #.......#
    # #########

    # randomized agent position and orientation
    agent_position = choice(
        rng,
        [
            position
            for position in grid.area.positions()
            if isinstance(grid[position], DeliverPoint)
        ],
    )
    agent_orientation = choice(rng, list(Orientation))
    agent = Agent(agent_position, agent_orientation)

    # remove coin from agent initial posiCtion
    grid[agent.position] = Floor()

    return State(grid, agent)


@transition_function_registry.register
def reload_items_from_Hub_transition( #reload
    state: State,
    action: Action,
    *,
    rng: Optional[rnd.Generator] = None,
):
    """if drone has enough capacity, then drone reload items from Hub"""
    if isinstance(state.grid[state.agent.position], Hub):
        if state.grid[state.agent.position].state_index == 0 and ( state.agent.capacity + Hub.item_num <= state.agent.max_capacity ): # OPEN
            state.agent.capacity += Hub.item_num
            state.grid[state.agent.position].set_closed() 
        print(state.agent.capacity)

@transition_function_registry.register
def unload_transition( #unload
    state: State,
    action: Action,
    *,
    rng: Optional[rnd.Generator] = None,
):
    """if drone has items, then drone unload items to Deliver Point"""
    if isinstance(state.grid[state.agent.position], DeliverPoint):
        if state.agent.capacity > 0 : 
            state.grid[state.agent.position] = Floor()
            state.agent.capacity -= 1
        print(state.agent.capacity)


@reward_function_registry.register
def finish_deliver_reward(
    state: State,
    action: Action,
    next_state: State,
    *,
    reward: float = 1.0,
    rng: Optional[rnd.Generator] = None,
):
    """gives reward if a delivery is correctly"""
    return (
        reward
        if isinstance(state.grid[next_state.agent.position], DeliverPoint)
        else 0.0
    )


@terminating_function_registry.register
def no_more_Deliverys(
    state: State,
    action: Action,
    next_state: State,
    *,
    rng: Optional[rnd.Generator] = None,
):
    """terminates episodes if all coins are collected"""
    return not any(
        isinstance(next_state.grid[position], DeliverPoint)
        for position in next_state.grid.area.positions()
    )
