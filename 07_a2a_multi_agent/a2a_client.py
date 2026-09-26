# the A2A protocol with NO agent framework: discover an agent, send it a message, read the reply.
# this is all any client (python, javascript, java, ...) needs to talk to an A2A agent.
#
# run (terminal 3, servers must be running):  python 07_a2a_multi_agent/a2a_client.py

import asyncio

import httpx
from a2a.client import A2ACardResolver, create_client
from a2a.helpers import get_stream_response_text, new_text_message
from a2a.types import SendMessageRequest

AGENTS = {"researcher": "http://localhost:8001", "writer": "http://localhost:8002"}


async def ask(url: str, text: str) -> str:
    async with httpx.AsyncClient() as http:            # 1. DISCOVER: download the agent card
        card = await A2ACardResolver(http, base_url=url).get_agent_card()
    print(f"\n--> {card.name}: {card.description}")
    client = await create_client(card)                 # 2. CONNECT using what the card says
    request = SendMessageRequest(message=new_text_message(text))
    answer = ""
    async for response in client.send_message(request):  # 3. SEND a message, stream the replies
        answer = get_stream_response_text(response) or answer
    return answer


async def main():
    facts = await ask(AGENTS["researcher"], "the latest langchain version")
    print(facts)
    paragraph = await ask(AGENTS["writer"], f"Facts:\n{facts}")
    print(paragraph)


if __name__ == "__main__":
    asyncio.run(main())
