import os
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor
from handle_multiprocessing import process_request
from chatbot_agent import ChatbotAgent


def main(message="", messages=[""]):
    # Initialize the ChatbotAgent.
    load_dotenv()
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set.")
    qdrant_url = os.getenv("QDRANT_URL")
    if not qdrant_url:
        raise ValueError("QDRANT_URL environment variable not set.")
    qdrant_api_key = os.getenv("QDRANT_API_KEY")
    if not qdrant_api_key:
        raise ValueError("QDRANT_API_KEY environment variable not set.")

    if message:
        messages.append(message)

    global chatbot_agent
    chatbot_agent = ChatbotAgent(
        openai_api_key=openai_api_key,
        qdrant_url=qdrant_url,
        qdrant_api_key=qdrant_api_key,
        messages=messages,
    )

    # Start the conversation.
    query = messages[-1]
    answer_list = []
    link_list = []
    # query it using content vector.
    query_results = chatbot_agent.search_context_qdrant(
        chatbot_agent.convert_chat_history_to_string(new_query=query, user_only=True),
        "Articles",
        top_k=4,
    )

    article_ids_plus_one = [min(article.id + 1, 945 - 1) for article in query_results]
    article_ids_minus_one = [max(article.id - 1, 0) for article in query_results]
    retrieved_articles_plus_one = chatbot_agent.client.retrieve(
        collection_name="Articles", ids=article_ids_plus_one
    )
    retrieved_articles_minus_one = chatbot_agent.client.retrieve(
        collection_name="Articles", ids=article_ids_minus_one
    )
    requests = [
        (
            chatbot_agent,
            # Concatenate the existing article content with the content retrieved using the article's id.
            # 'retrieve' function returns a list of points, so we need to access the first (and in this case, only) result with '[0]'.
            (retrieved_articles_minus_one[i].payload["content"] + "\n" +
             article.payload["content"] + "\n" +
             retrieved_articles_plus_one[i].payload["content"]),
            chatbot_agent.convert_chat_history_to_string(
                user_only=True, remove_resource=True
            ),
            query,
            article.payload["link"],
            article.score,
        )
        for i, article in enumerate(query_results)
    ]

    # Use a Pool to manage the processes.
    with ThreadPoolExecutor(max_workers=len(query_results)) as executor:
        results = list(executor.map(process_request, requests))

    # Results is a list of tuples of the form (answer, link).
    answer_list, link_list = zip(*results)

    # Initialize link_list_list with each link from link_list as a separate list.
    link_list_list = [[link] for link in link_list]
    # For each answer, perform the query and add the result to the corresponding list in link_list_list.
    for i, answer in enumerate(answer_list):
        secondary_query_results_temp = chatbot_agent.search_context_qdrant(
            answer, "Articles", top_k=2
        )
        link_list_list[i].extend(
            article.payload["link"].replace("_sources", "").replace(".md", ".html")
            for article in secondary_query_results_temp
        )

    combine_answer = chatbot_agent.prompt_combine_chain(
        query=query, answer_list=answer_list, link_list_list=link_list_list
    )

    return combine_answer


if __name__ == "__main__":
    main()
