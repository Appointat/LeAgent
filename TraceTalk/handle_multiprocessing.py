# Define the process_request function.
def process_request(params):
    """
    Process a request to the chatbot.

    Args:
        params (tuple): A tuple of parameters.

    Returns:
        tuple: A tuple of the answer and the link.
    """
    chatbot_agent, context, chat_history, query, link, score = params
    convert_link = (
        link.replace("_sources", "").replace(".md", ".html") if score > 0.5 else ""
    )
    reject_context = "Sorry, the question is not associate with the context. The chatbot should refuse to answer."

    try:
        chain = chatbot_agent.prompt_chatbot()
        answer = (
            chain.predict(
                context=context,
                chat_history=chat_history,
                query=query,
            )
            if score > 0.5
            else reject_context
        )
    finally:
        # Release resources here, for example:
        # chatbot_agent.close()
        pass
    return answer, convert_link
