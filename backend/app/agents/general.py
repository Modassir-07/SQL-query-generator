from app.state.state import AgentState
from app.services.llm import get_llm_client, GEMINI_MODEL

async def general_agent_node(state: AgentState):
    """
    Handles queries that are not relevant to the specific database domain.
    Provides a helpful response and guides the user back to the supported topics.
    """
    client = get_llm_client()
    user_query = state["user_query"]
    
    system_prompt = """You are a helpful assistant for a Distribution Analytics application based on the Chinook music store database.
    The user has asked a question that is outside the scope of this database.
    
    Your task is to:
    1. Politely acknowledge the user's query.
    2. Explain that you specialize in analyzing the Chinook music store data (Sales, Customers, Tracks, Artists, Invoices).
    3. Suggest 2-3 relevant questions they could ask instead.
    
    Keep the tone professional and helpful.
    """
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_query}
    ]

    # Use LangGraph's native stream writer
    from langgraph.types import StreamWriter
    from langgraph.config import get_stream_writer

    writer: StreamWriter = get_stream_writer()

    import asyncio

    print("[General Agent] Requesting completion from Gemini...")
    full_response = ""
    try:
        stream = await asyncio.wait_for(
            client.chat.completions.create(
                model=GEMINI_MODEL,
                reasoning_effort="low",
                messages=messages,
                temperature=0.7,
                stream=True
            ),
            timeout=30
        )

        async def _consume():
            nonlocal full_response
            async for chunk in stream:
                if not chunk.choices:
                    continue
                content = chunk.choices[0].delta.content
                if content:
                    full_response += content
                    writer(content)

        await asyncio.wait_for(_consume(), timeout=60)
        print(f"[General Agent] Done. {len(full_response)} chars received.")
    except asyncio.TimeoutError:
        print("[General Agent] TIMED OUT waiting on Gemini stream.")
        full_response = "Sorry, the response is taking too long to generate. Please try again."
    except Exception as e:
        print(f"[General Agent] ERROR: {type(e).__name__}: {e}")
        full_response = f"Sorry, something went wrong while generating the response: {e}"

    return {"natural_response": full_response}
