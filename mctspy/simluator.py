import abc
import typing as t


class MCTSSimulator(abc.ABC):
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
    def get_initial_state(self) -> t.Hashable:
        """ Get initial state.
        """
        pass

    @abc.abstractmethod
    def get_agent_num(self) -> int:
        """ Get the number of agents (players).
        """
        pass

    @abc.abstractmethod
    def get_terminal_value(self, state: t.Hashable) -> t.Dict[t.Hashable, float]:
        """ Get the value of the terminal state for every agent.
        """
        pass