# Define the process_request function.
def process_request(params):
    chatbot_agent, context, chat_history, summaries, query, link, score = params
    convert_link = link.replace('_sources', '').replace('.md', '.html') if score > 0.715 else ""
    reject_context = "sorry, the question is not associate with the context. The chatbot should refuse to answer."
    
    try:
        chain = chatbot_agent.prompt_chatbot()
        answer = chain.predict(
            context=context,
            chat_history=chat_history,
            summaries=summaries,
            query=query,
        ) if score > 0.715 else reject_context
    finally:
        # Release resources here, for example:
        # chatbot_agent.close()
        pass
    return answer, convert_link