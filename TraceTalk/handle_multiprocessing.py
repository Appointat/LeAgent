import re

def process_request(params):
    """
    Process a request to the chatbot.

    Args:
        params (tuple): A tuple of parameters.

    Returns:
        tuple: A tuple of the answer and the link.
    """
    chatbot_agent, context, chat_history, query, link, score = params
    convert_link = link.replace("_sources", "").replace(".md", ".html") if score > 0.5 else ""
    reject_context = "Sorry, the question is not associated with the context. The chatbot should refuse to answer."

    url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+|www.(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+|[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}'
    resource_list = re.findall(url_pattern, context)
    resource_str = "\n".join([f"[{i+1}] {link}" for i, link in enumerate(resource_list)])
    resource_str = convert_link + "\n" + resource_str

    try:
        if score > 0.5:
            answer = chatbot_agent.prompt_chatbot(context, chat_history, resource_str, query)
        else:
            answer = reject_context
    except Exception as e:
        print(f"An error occurred: {e}")
        answer = "I'm sorry, but I encountered an error while processing your request."
    finally:
        # Release resources here, for example:
        # chatbot_agent.close()
        pass

    return answer, convert_link