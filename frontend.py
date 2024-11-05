import chainlit as cl 
from main import RAGAgent

@cl.on_message
async def main(message: cl.Message):
    print(f"Received: {message.content}")
    agent = RAGAgent()
    result = agent.run(message.content)
    await cl.Message(
        content=f"Result: {result}",
    ).send()
# Which types of proceedings are excluded from the application of this Act as per Section 3?