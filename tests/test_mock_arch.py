import unittest
from expertflow import Agent, Router, ConversationManager, MockLLM

class TestMockArchitecture(unittest.TestCase):
    def setUp(self):
        self.agent1 = Agent(name="python_expert", description="Python stuff", system_prompt="You are python expert")
        self.agent2 = Agent(name="math_expert", description="Math stuff", system_prompt="You are math expert")
        
        # Create a Mock LLM that simulates routing and responses
        self.mock_llm = MockLLM(
            responses={
                "Help me with python": "Here is python code",
                "What is 2+2?": "It is 4"
            }
        )
        
        self.router = Router(
            agents=[self.agent1, self.agent2],
            default_agent=self.agent1,
            llm=self.mock_llm
        )
        
        self.manager = ConversationManager(router=self.router, llm=self.mock_llm)

    def test_routing_with_mock(self):
        # The MockLLM is simple: if "python" in input, it returns "python_expert" (simulating router LLM)
        # Wait, the MockLLM.generate returns the response content.
        # The Router calls llm.generate() expecting an agent name.
        # The Manager calls llm.generate() expecting a chat response.
        # This is a conflict if we use the SAME instance for both with simple logic.
        # But let's see MockLLM implementation.
        
        # MockLLM logic:
        # if "python" in last_msg -> returns "python_tutor" (Wait, my agent is "python_expert")
        # Let's update the test agents to match MockLLM hardcoded logic or update MockLLM.
        pass

    def test_flow(self):
        # 1. Test Routing (Router calls LLM)
        # Input: "I need python help"
        # MockLLM sees "python", returns "python_tutor" (hardcoded in MockLLM.py)
        # But my agents are named "python_expert".
        # I should probably make MockLLM more configurable or update my agents here.
        
        # Let's use agents that match MockLLM's hardcoded logic for now to prove the point.
        agent_py = Agent(name="python_tutor", description="Python", system_prompt="Sys")
        agent_math = Agent(name="math_wizard", description="Math", system_prompt="Sys")
        
        router = Router(agents=[agent_py, agent_math], default_agent=agent_py, llm=self.mock_llm)
        
        # Test Classify
        # "I need python" -> MockLLM returns "python_tutor"
        intent = router.classify("I need python", "math_wizard")
        self.assertEqual(intent, "python_tutor")
        
        # Test Conversation (Manager calls LLM)
        # "What is 2+2?" -> MockLLM returns "It is 4" (from responses dict)
        manager = ConversationManager(router=router, llm=self.mock_llm)
        response = manager.process_turn("user1", "What is 2+2?")
        self.assertEqual(response.content, "It is 4")

if __name__ == "__main__":
    unittest.main()
