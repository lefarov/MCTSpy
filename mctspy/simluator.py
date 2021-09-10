import abc
import typing as t


class SimulatorInterface(abc.ABC):
    """ Multiagent simulator interface supported by MCST.
    """
    @abc.abstractmethod
    def step(
        self, state: t.Hashable, action: t.Hashable
    ) -> t.Tuple[t.Hashable, float, t.Hashable]:
        """ Step through simulation.
        """
        pass

    @abc.abstractmethod
    def state_is_terminal(self, state: t.Hashable) -> bool:
        """ Check if state is terminal.
        """
        pass

    @abc.abstractmethod
    def enumerate_actions(self, state: t.Hashable) -> t.Set:
        """ Enumerate all possivle actions for given state.
        """
        pass
    
    @abc.abstractmethod
    def get_initial_state(self) -> t.Tuple[t.Hashable, t.Hashable]:
        """ Get initial state and the ID of an agent who is taking action.
        """
        pass

    @abc.abstractmethod
    def get_agent_num(self) -> int:
        """ Get the number of agents (players).
        """
        pass

    @abc.abstractmethod
    def get_current_agent(self, state: t.Hashable) -> t.Hashable:
        """ Get the ID of an agent who supposed to take action.
        """
        pass

    @abc.abstractmethod
    def get_terminal_value(self, state: t.Hashable) -> t.Dict[t.Hashable, float]:
        """ Get the value of the terminal state for every agent.
        """
        pass


class SimulatorInterfacePO(SimulatorInterface):
    """Simulator Interface for Partiall-observed simulations.
    
    TODO:
    1. Should we keep observation as separate function?
    """

    @abc.abstractmethod
    def step(
        self, state: t.Hashable, action: t.Hashable
    ) -> t.Tuple[t.Hashable, t.Hashable, float, t.Hashable]:
        """ Step through simulation.

        Returns
        -------
        next_state: hashable
        next_observations: hashable
        reward: float
        next_agent_id: hashable
        """
        pass

    @abc.abstractmethod
    def get_initial_state(self) -> t.Tuple[t.Hashable, t.Hashable, t.Hashable]:
        """ Get initial state and the ID of an agent who is taking action.

        Returns
        -------
        initial_state: hashable
        initial_observation: hashable
        agent_id: hashable
        """
        pass