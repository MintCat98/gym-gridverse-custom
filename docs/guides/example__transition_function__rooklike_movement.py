from typing import Optional

import numpy.random as rnd

from gym_gridverse.actions import Actions
from gym_gridverse.state import State

# mapping movement actions to lambda functions which map the agent's
# orientation to a direction of movement
action_to_orientation = {
    Actions.MOVE_FORWARD: lambda ori: ori,
    Actions.MOVE_LEFT: lambda ori: ori.rotate_left(),
    Actions.MOVE_RIGHT: lambda ori: ori.rotate_right(),
    Actions.MOVE_BACKWARD: lambda ori: ori.rotate_back(),
}


def rooklike_movement(
    state: State,
    action: Actions,
    *,
    rng: Optional[rnd.Generator] = None,  # pylint: disable=unused-argument
) -> None:
    """moves the agent in a rooklike fashion, until it hits an obstacle"""

    try:
        orientation_to_position_delta = action_to_orientation[action]
    except KeyError:  # not a movement action
        return

    # agent's movement direction as a position
    position_delta = orientation_to_position_delta(
        state.agent.orientation
    ).as_position()

    # move agent until it hits a blocking cell
    while True:
        next_position = state.agent.position + position_delta

        if state.grid[next_position].blocks:
            break

        state.agent.position = next_position
