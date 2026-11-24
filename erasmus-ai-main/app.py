from dotenv import load_dotenv

load_dotenv()

import chainlit as cl

from graph.agent import build_graph

graph = build_graph()


@cl.on_chat_start
async def on_chat_start():
    cl.user_session.set("history", [])


@cl.on_message
async def on_message(message: cl.Message):
    history = cl.user_session.get("history", [])

    state = {
        "history": history,
        "user_input": message.content,
    }

    result = await graph.ainvoke(state)

    history.append({"role": "user", "content": message.content})
    history.append({"role": "assistant", "content": result["final_answer"]})
    cl.user_session.set("history", history)

    await cl.Message(content=result["final_answer"]).send()
