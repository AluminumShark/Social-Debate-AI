from agents.agent_a import AgentA
from agents.agent_b import AgentB
from agents.agent_c import AgentC
from utils.config_loader import Config
import textwrap

class DialogueManager:
    def __init__(self):
        cfg = Config.load()
        self.rounds = cfg.get('debate', {}).get('rounds', 3)
        self.topic = cfg.get('debate', {}).get('topic', 'Should we adopt universal basic income?')
        self.agents = {
            'A': AgentA(name='A', config=cfg),
            'B': AgentB(name='B', config=cfg),
            'C': AgentC(name='C', config=cfg),
        }
        self.state = {
            'last_message': self.topic,
            'topic': self.topic,
            'history': []
        }

    def run(self):
        bar = "─" * 60
        print(f"\n{bar}\nTopic: {self.topic}\n{bar}")

        for rnd in range(1, self.rounds + 1):
            print(f"\nRound {rnd}\n{bar}")
            for agent in self.agents.values():
                reply = agent.select_action(self.state)

                # Wrap long text
                wrapped = textwrap.fill(reply, width=80, subsequent_indent=" " * 9)
                print(f"{agent.name:>6} │ {wrapped}")

                # Update histories
                for ag in self.agents.values():
                    ag.update_history(reply)

                self.state["last_message"] = reply
                self.state["history"].append({"speaker": agent.name, "text": reply})

        print(f"\n{bar}\nDebate Complete\n{bar}")