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
from gym_gridverse.grid_object import (
    Color,
    Floor,
    GridObject,
    Wall,
    DeliveryAddress,
    DeliveryHub,
)
from gym_gridverse.rng import choice, get_gv_rng_if_none
from gym_gridverse.state import State


@reset_function_registry.register
def coin_maze(*, rng: Optional[rnd.Generator] = None) -> State:
    """creates a maze with collectible coins"""

    # must call this to include reproduceable stochasticity
    rng = get_gv_rng_if_none(rng)

    # initializes grid with Coin
    grid = Grid.from_shape((7, 9), factory=Floor)
    # assigns Wall to the border
    draw_wall_boundary(grid)
    # draw other walls
    #draw_room(grid, Area((2, 4), (2, 6)), DeliveryHub)
    # re-assign openings
    grid[2, 3] = DeliveryHub()
    grid[5, 4] = DeliveryAddress(3)
    grid[3, 4] = DeliveryAddress(3)
    grid[3, 5] = DeliveryAddress(2)
    grid[3, 6] = DeliveryAddress(2)
    grid[3, 7] = DeliveryAddress(1)
    #grid[3, 8] = DeliveryAddress()

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
            if isinstance(grid[position], Floor)
        ],
    )
    agent_orientation = choice(rng, list(Orientation))
    agent = Agent(agent_position, agent_orientation)

    # remove coin from agent initial posiCtion
    grid[agent.position] = Floor()

    return State(grid, agent)


@transition_function_registry.register
def reload_items_from_Hub_transition(  # reload
    state: State,
    action: Action,
    *,
    rng: Optional[rnd.Generator] = None,
):
    """if drone has enough capacity, then drone reload items from Hub"""
    if isinstance(state.grid[state.agent.position], DeliveryHub) and action == Action.ACTUATE:
        Hub = state.grid[state.agent.position] 
        if state.grid[state.agent.position].state_index == 0 and (Hub.is_empty == False):  # OPEN
            if state.agent.max_capacity > state.agent.capacity :
                state.agent.capacity += 1
                Hub.item_num -= 1
        print(f'drone capacity after reload : {state.agent.capacity}')

        if Hub.is_empty :
            Hub.state_index = 1
            print(f'Hub is empty')

        



@transition_function_registry.register
def unload_transition(  # unload
    state: State,
    action: Action,
    *,
    rng: Optional[rnd.Generator] = None,
):
    """if drone has items, deliveryAddress has req items, then drone unload one item to DeliverAddress"""
    if isinstance(state.grid[state.agent.position], DeliveryAddress) and action == Action.ACTUATE:
        delivery_address = state.grid[state.agent.position] 
        if state.agent.capacity > 0 and delivery_address.is_empty == False:
            print(f'delivery before : {delivery_address.num_items}')
            delivery_address.num_items -= 1
            state.agent.finished_deliver_num += 1
            state.agent.capacity -= 1
            print(f'delivery after : {delivery_address.num_items}')
        print(f'drone capacity after unload : {state.agent.capacity}')


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
    if isinstance(state.grid[next_state.agent.position], DeliveryAddress) and action == Action.ACTUATE :
        DeliverAdd = state.grid[next_state.agent.position]
        if (DeliverAdd.is_empty != True) and state.agent.capacity > 0:
            reward = 10.0
        elif state.agent.capacity == 0:
            reward = -1.5
        elif DeliverAdd.is_empty == True :
            reward = -1.5
        else :
            reward = 0
    else :
        reward = 0
    print(f"finish_deliver reward : {reward}")
 
    return reward

@reward_function_registry.register
def reload_reward(
    state: State,
    action: Action,
    next_state: State,
    *,
    reward: float = 1.0,
    rng: Optional[rnd.Generator] = None,
):
    """gives reward if a delivery is correctly"""   
    if isinstance(next_state.grid[next_state.agent.position], DeliveryHub) and action == Action.ACTUATE:
        Hub = state.grid[state.agent.position] 
        if state.grid[state.agent.position].state_index == 0 and (Hub.is_empty == False):  # OPEN
            if state.agent.max_capacity > state.agent.capacity :
                reward = 10.0
            else :
                reward = 0
        if (Hub.is_empty == True) and (Hub.is_rewarded == False):
            reward = 5
            Hub.is_rewarded = True
        else :
            reward = 0
    else :
        reward = 0
    print("reload_reward : ")
    print(reward)

    return reward

@reward_function_registry.register
def actuate_on_empty(
    state: State,
    action: Action,
    next_state: State,
    *,
    reward: float = 1.0,
    rng: Optional[rnd.Generator] = None,
):
    """gives reward if a delivery is correctly"""   
    if  action == Action.ACTUATE :
        if isinstance(state.grid[next_state.agent.position], DeliveryAddress) or isinstance(state.grid[next_state.agent.position], DeliveryHub):
            Obj = state.grid[state.agent.position]
            if (Obj.is_empty == True) or (state.agent.capacity == 0):
                reward = -10
            else :
                reward = 10 
        else :
            reward = -5
    else :
        reward = 0
    print(f"actuate_on_empty reward : {reward}")
 
    return reward

@reward_function_registry.register
def holding_penalty(
    state: State,
    action: Action,
    next_state: State,
    *,
    reward: float = -0.2,
    rng: Optional[rnd.Generator] = None,
):
    """gives reward if a delivery is correctly"""
    print("holding penalty reward : ")
    print(reward * state.agent.capacity
        if state.agent.capacity > 0
        else 0.0)
    return (
        reward * state.agent.capacity
        if state.agent.capacity > 0
        else 0.0
    )
    

@reward_function_registry.register
def terminating_reward(
    state: State,
    action: Action,
    next_state: State,
    *,
    reward: float = 50,
    rng: Optional[rnd.Generator] = None,
):
    """gives reward if all delivery is finish """
    print("terminating_reward : ")
    print(reward
        if DeliveryHub.target_num == next_state.agent.finished_deliver_num
        else 0.0)
    return (
        reward
        if DeliveryHub.target_num == next_state.agent.finished_deliver_num
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
    return DeliveryHub.target_num == next_state.agent.finished_deliver_num
    