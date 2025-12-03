import unittest
from unittest.mock import MagicMock, patch
import sys

# Ensure we can import the package
sys.path.append(".")

from expertflow import Agent, Router, ConversationManager, Message

class TestAgents(unittest.TestCase):
    def setUp(self):
        self.agent1 = Agent(
            name="math_expert",
            description="Expert in math",
            system_prompt="You are a math expert."
        )
        self.agent2 = Agent(
            name="python_expert",
            description="Expert in python",
            system_prompt="You are a python expert."
        )
        self.agents = [self.agent1, self.agent2]

    def test_agent_initialization(self):
        agent = Agent(
            name="test",
            description="desc",
            system_prompt="prompt"
        )
        self.assertEqual(agent.name, "test")
        self.assertEqual(agent.description, "desc")
        self.assertEqual(agent.system_prompt, "prompt")
        self.assertEqual(agent.model_name, "gemini-2.0-flash") # Default

    def test_router_classification(self):
        # Setup mock LLM
        mock_llm = MagicMock()
        mock_llm.generate.return_value = "python_expert"

        router = Router(agents=self.agents, default_agent=self.agent1, llm=mock_llm)
        
        # Test classification
        result = router.classify("Help me with python code", "math_expert")
        
        self.assertEqual(result, "python_expert")
        mock_llm.generate.assert_called_once()

    def test_conversation_flow(self):
        # Setup mock LLM
        mock_llm = MagicMock()
        
        # 1. Router call: returns "python_expert"
        # 2. Manager call: returns "Here is some python code."
        mock_llm.generate.side_effect = ["python_expert", "Here is some python code."]
        mock_llm.get_token_usage.return_value = {"total": 10}

        # Initialize
        router = Router(agents=self.agents, default_agent=self.agent1, llm=mock_llm)
        manager = ConversationManager(router=router, llm=mock_llm)

        # Test process_turn
        response = manager.process_turn("user1", "Write a python script")

        # Verify switch happened
        self.assertTrue(response.switched_context)
        self.assertEqual(response.agent_name, "python_expert")
        self.assertEqual(response.content, "Here is some python code.")
        
        # Verify history update
        session_data = manager._get_session("user1")
        self.assertEqual(len(session_data["history"]), 2) # User msg + Assistant msg
        self.assertEqual(session_data["history"][0].content, "Write a python script")
        self.assertEqual(session_data["history"][1].content, "Here is some python code.")

    def test_no_switch_flow(self):
        # Setup mock LLM
        mock_llm = MagicMock()
        
        # 1. Router call: returns "math_expert" (same as default)
        # 2. Manager call: returns "2 + 2 is 4."
        mock_llm.generate.side_effect = ["math_expert", "2 + 2 is 4."]
        mock_llm.get_token_usage.return_value = {"total": 10}

        # Initialize
        router = Router(agents=self.agents, default_agent=self.agent1, llm=mock_llm)
        manager = ConversationManager(router=router, llm=mock_llm)

        # Test process_turn
        response = manager.process_turn("user1", "What is 2+2?")

        # Verify NO switch
        self.assertFalse(response.switched_context)
        self.assertEqual(response.agent_name, "math_expert")
        self.assertEqual(response.content, "2 + 2 is 4.")

if __name__ == "__main__":
    unittest.main()
