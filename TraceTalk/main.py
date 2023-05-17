import os
from chatbot_agent import ChatbotAgent



def main():
	# Set the OpenAI API key.
	with open(r'TraceTalk\OpenAI-API-key\OpenAI_API_key.txt', 'r') as f:
		api_key = f.read().strip()
	os.environ['OPENAI_API_KEY'] = api_key
	print("Set the OpenAI API key.\n")

	# Initialize the ChatbotAgent.
	chatbot_agent = ChatbotAgent(openai_api_key=os.environ["OPENAI_API_KEY"])
	
	# Start the conversation.
	print("\n\n****Chatbot Agent Initialized****")
	while chatbot_agent.count <= 20:
		print("[{}]".format(chatbot_agent.count))
		query = input("Query   : ")
		answer_list = []
		link_list = []
		# query it using content vector.
		query_results = chatbot_agent.search_context_qdrant(query, 'Articles', top_k=4)
		for i, article in enumerate(query_results):
			print(f'{i + 1}. {article.payload["title"]} (Score: {round(article.score, 3)}), link: {article.payload["link"]}')
			chain = chatbot_agent.prompt_the_chatbot()
			answer = chain.predict(
				context=article.payload["content"], 
				chat_history=chatbot_agent.convert_chat_history_to_string(), 
				summaries="",
				qury=query,
			)
			answer_list.append(answer)
			link_list.append(article.payload["link"])

		n = len(answer_list)
		if n == 0:
			combine_answer = "I'm sorry, there is not enough information to provide a meaningful answer to your question. Can you please provide more context or a specific question?"
		else:
			context=f"""
Now I will provide you with {n} chains, here is the definition of chain: each chain contains an answer and a link. The answers in the chain are the results from the links.
In theory, each chain should produce a paragraph with MD links as references. It means that you MUST tell me from which references you make the summery.
The smaller the number of the chain, the more important the information contained in the chain.
But if the meaning of an answer in a certain chain is similar to 'I am not sure about your question' or 'I refuse to answer such a question', it means that this answer chain is deprecated, and you should actively ignore the information in this answer chain.

You now are asked to COMBINE these {n} chains (combination means avoiding repetition, smooth writing, giving verbose answer).
The finl answer is ALWAYS in the form of TEXT WITH MD LINK. If no refernce for one sentence, you do not need to attach the link to that sentence.
In addition, ALWAYS return "TEXT WITH MD LINK", and ALSO ALWAYs return a "REFERENCE" part in your answer (they are two parts).
For exmaple:
    I provide the input text:
		CHAIN 1:
			CONTEXT: 
				Machine learning algorithms build a model based on sample data, known as training data, in order to make predictions or decisions without being explicitly programmed to do so.
			SOURCES: 
				Blabla.
			REFERENCE: 
				https://en.wikipedia.org/wiki/Machine_learning
		CHAIN 2:
			TEXT: A convolutional neural network (CNN) is a type of artificial neural network commonly used in deep learning for image recognition and classification. 
			REFERENCE: https://open-academy.github.io/machine-learning/_sources/deep-learning/image-classification.md
    Your output should be:
		COMBINATION: 
			Machine learning is a method of teaching computers to learn patterns in data without being explicitly programmed. It involves building models that can make predictions or decisions based on input data [1]. 
			One type of machine learning model commonly used for sequential or time series data is recurrent neural networks (RNNs) [2]. 
        REFERENCE: 
			[1] https://en.wikipedia.org/wiki/Machine_learning
			[2] https://open-academy.github.io/machine-learning/_sources/deep-learning/image-classification.md
=========
The user's query is: '{query}'.

=========
        """
			for i in range(n):
				context += f"""
### CHAIN {i+1}
CONTEXT: 
	{answer_list[i]}
REFERENCE: 
	{link_list[i]}
"""
			combine_chain = chatbot_agent.combine_prompt()
			combine_answer = combine_chain.run(
				context = context,
			)
		print(f'Answer: {combine_answer}\n')
		chatbot_agent.update_chat_history(query, combine_answer)


if __name__ == "__main__":
	main()