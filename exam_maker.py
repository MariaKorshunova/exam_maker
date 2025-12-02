import os
from dotenv import find_dotenv, load_dotenv, dotenv_values

from langchain_gigachat.chat_models import GigaChat
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

load_dotenv(find_dotenv())
env_file_values = dotenv_values(find_dotenv())

os.environ["GIGACHAT_CREDENTIALS"] = env_file_values.get("GIGACHAT_CREDENTIALS")

model = GigaChat(
    model="GigaChat-2-Max",
    verify_ssl_certs=False,
)

system_prompt = ""

agent = create_react_agent(model,
                           tools=[],
                           checkpointer=MemorySaver(),
                           prompt=system_prompt)

rq = "Создай 3 тестовых вопроса по теме Машинное обучение. В ответе должно быть checkbox."

config = {"configurable": {"thread_id": 42}}
resp = agent.invoke({"messages": [("user", rq)]}, config=config)
print(resp["messages"][-1].content)
