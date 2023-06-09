# Define the process_request function.
def process_request(params):
    chatbot_agent, context, chat_history, summaries, query, link = params
    convert_link = link.replace('_sources', '').replace('.md', '.html')
    try:
        chain = chatbot_agent.prompt_chatbot()
        answer = chain.predict(
            context=context,
            chat_history=chat_history,
            summaries=summaries,
            qury=query,
        )
    finally:
        # Release resources here, for example:
        # chatbot_agent.close()
        pass
    return answer, convert_link