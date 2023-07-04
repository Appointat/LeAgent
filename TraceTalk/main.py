import os
from enum import Enum
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor
from handle_multiprocessing import process_request
from chatbot_agent import ChatbotAgent



class Role(Enum):
    ASSISTANT = "assistant"
    USER = "user"



class Message:
    def __init__(self, role: Role, content: str):
        self.role = role
        self.content = content



def main(message="", messages=[""]):
	# Initialize the ChatbotAgent.
	load_dotenv()
	openai_api_key = os.getenv('OPENAI_API_KEY')
	if not openai_api_key:
		raise ValueError("OPENAI_API_KEY environment variable not set.")
	qdrant_url = os.getenv('QDRANT_URL')
	if not qdrant_url:
		raise ValueError("QDRANT_URL environment variable not set.")
	qdrant_api_key = os.getenv('QDRANT_API_KEY')
	if not qdrant_api_key:
		raise ValueError("QDRANT_API_KEY environment variable not set.")
	

	use_REST_API = False	
	if message or messages:
		use_REST_API = True
		message = messages.pop(-1)
	messages.append(message)


	global chatbot_agent
	chatbot_agent = ChatbotAgent(openai_api_key=openai_api_key, qdrant_url=qdrant_url, qdrant_api_key=qdrant_api_key, messages=messages)


	# Start the conversation.
	if not message and not use_REST_API:
		print("\n\n****Chatbot Agent Initialized****")
		while chatbot_agent.count <= 10:
			print("[{}]".format(chatbot_agent.count))
			query = input("Query   : ")
			answer_list = []
			link_list = []
			# Query it using content vector.
			query_results = chatbot_agent.search_context_qdrant(chatbot_agent.convert_chat_history_to_string(user_only=True)+"\nuser: "+query, 'Articles', top_k=4)

			article_ids_plus_one = [article.id + 1 for article in query_results]
			article_ids_minus_one = [max(article.id - 1, 0) for article in query_results]
			retrieved_articles_plus_one = chatbot_agent.client.retrieve(collection_name="Articles", ids=article_ids_plus_one)
			retrieved_articles_minus_one = chatbot_agent.client.retrieve(collection_name="Articles", ids=article_ids_minus_one)
			requests = [
				(
					chatbot_agent,
					# Concatenate the existing article content with the content retrieved using the article's id.
					# 'retrieve' function returns a list of points, so we need to access the first (and in this case, only) result with '[0]'.
					retrieved_articles_minus_one[i].payload["content"] + "\n" +
					article.payload["content"]
					+ "\n" + retrieved_articles_plus_one[i].payload["content"],
					# Convert the chat history to string, including only user's side of the chat (user_only=True).
					chatbot_agent.convert_chat_history_to_string(user_only=True),
					"",
					query,
					article.payload["link"],
					article.score
				) 
				for i, article in enumerate(query_results)  # Looping over each article in the query results.
			]

			# Use a Pool to manage the processes.
			with ThreadPoolExecutor(max_workers=len(query_results)) as executor:
				results = list(executor.map(process_request, requests))

			# Results is a list of tuples of the form (answer, link).
			answer_list, link_list = zip(*results)

			combine_answer = chatbot_agent.prompt_combine_chain(query=query, answer_list=answer_list, link_list=link_list)
			print(f'Query : {query}\n')
			print(f'Answer: {combine_answer}\n')
			chatbot_agent.update_chat_history(query, combine_answer)
	else:
		query = message
		answer_list = []
		link_list = []
		# query it using content vector.
		query_results = chatbot_agent.search_context_qdrant(chatbot_agent.convert_chat_history_to_string(user_only=True)+"\n[user]: "+query, 'Articles', top_k=4)

		article_ids_plus_one = [min(article.id + 1, 391 - 1) for article in query_results]
		article_ids_minus_one = [max(article.id - 1, 0) for article in query_results]
		retrieved_articles_plus_one = chatbot_agent.client.retrieve(collection_name="Articles", ids=article_ids_plus_one)
		retrieved_articles_minus_one = chatbot_agent.client.retrieve(collection_name="Articles", ids=article_ids_minus_one)
		requests = [
			(
				chatbot_agent,
				# Concatenate the existing article content with the content retrieved using the article's id.
				# 'retrieve' function returns a list of points, so we need to access the first (and in this case, only) result with '[0]'.
				retrieved_articles_minus_one[i].payload["content"] + "\n" +
				article.payload["content"]
				+ "\n" + retrieved_articles_plus_one[i].payload["content"],
				chatbot_agent.convert_chat_history_to_string(user_only=True),
				"",
				query,
				article.payload["link"],
				article.score
			) 
			for i, article in enumerate(query_results)
		]

		# Use a Pool to manage the processes.
		with ThreadPoolExecutor(max_workers=len(query_results)) as executor:
			results = list(executor.map(process_request, requests))

		# Results is a list of tuples of the form (answer, link).
		answer_list, link_list = zip(*results)

		combine_answer = chatbot_agent.prompt_combine_chain(query=query, answer_list=answer_list, link_list=link_list)

		return combine_answer



if __name__ == "__main__":
	main()