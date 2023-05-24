from langchain.chat_models import AzureChatOpenAI
from langchain.callbacks import get_openai_callback
from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.agents import AgentType
import os

llm = AzureChatOpenAI(
    openai_api_key=os.environ["OPENAI_API_KEY"], temperature=0.9,
    openai_api_base="https://yuchanns-openai.openai.azure.com",
    deployment_name="gpt35",
    openai_api_version="2023-03-15-preview",
    client=None
)

tools = load_tools(
    ["google-search", "llm-math"], llm=llm,
    google_api_key=os.environ["GOOGLE_API_KEY_"],
    google_cse_id=os.environ["GOOGLE_CSE_ID_"]
)

agent = initialize_agent(
    tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True,
)

with get_openai_callback() as cb:
    content = agent.run(
        "What is the best place to travel in Japan, October?"
    )
    print("successful_requests: ", cb.successful_requests)
    print("total_tokens: ", cb.total_tokens)
    print("prompt_tokens: ", cb.prompt_tokens)
    print("completion_tokens: ", cb.completion_tokens)
    print("total_cost: ", cb.total_cost)
    print("content: ", content)
