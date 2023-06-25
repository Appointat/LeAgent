import os
import sys
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor
from handle_multiprocessing import process_request
from chatbot_agent import ChatbotAgent



def main(message=[]):
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
	
	
	messages = []
	if not message and len(sys.argv) > 1:
		message = sys.argv[1]
	messages.append(message)


	global chatbot_agent
	chatbot_agent = ChatbotAgent(openai_api_key=openai_api_key, qdrant_url=qdrant_url, qdrant_api_key=qdrant_api_key, messages=messages)


	# Start the conversation.
	if not message:
		print("\n\n****Chatbot Agent Initialized****")
		while chatbot_agent.count <= 10:
			print("[{}]".format(chatbot_agent.count))
			query = input("Query   : ")
			answer_list = []
			link_list = []
			# Query it using content vector.
			query_results = chatbot_agent.search_context_qdrant(chatbot_agent.convert_chat_history_to_string()+"\nuser: "+query, 'Articles', top_k=4)
			requests = [(chatbot_agent, article.payload["content"], chatbot_agent.convert_chat_history_to_string(), "", query, article.payload["link"]) for article in query_results]

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
		query_results = chatbot_agent.search_context_qdrant(chatbot_agent.convert_chat_history_to_string()+"\nuser: "+query, 'Articles', top_k=4)
		requests = [(chatbot_agent, article.payload["content"], chatbot_agent.convert_chat_history_to_string(), "", query, article.payload["link"]) for article in query_results]

		# Use a Pool to manage the processes.
		with ThreadPoolExecutor(max_workers=len(query_results)) as executor:
			results = list(executor.map(process_request, requests))

		# Results is a list of tuples of the form (answer, link).
		answer_list, link_list = zip(*results)

		combine_answer = chatbot_agent.prompt_combine_chain(query=query, answer_list=answer_list, link_list=link_list)
		print(f'Answer: {combine_answer}\n')
		chatbot_agent.update_chat_history(query, combine_answer)


if __name__ == "__main__":
	main()