from langchain.prompts import PromptTemplate


# Prompt the chatbot.
def basic_prompt():
    template = """
CONTEXT information is below:
{{context}}
=========
Chat_history:
{{chat_history}}
=========
Given the following extracted parts of a long document and a QUESTION, create a final answer with references ("SOURCES", the refernces do not include links).
If you don't know the answer, just say that you don't know. Don't try to make up an answer. 
If the QUESTION is not associate with the CONTEXT, you can say "SORRY" and say that you need more information to answer it, or you can enven refuse to answer it.
ALWAYS return a "SOURCES" part in your answer.

QUESTION: {{query}}
=========
{{summaries}}
=========
FINAL ANSWER THE QUESTION {{query}}, language used for answers is CONSISTENT with QUESTION:
"""
    prompt = PromptTemplate(
        template=template,
        input_variables=["context", "chat_history", "summaries", "query"],
        template_format="jinja2",
        validate_template=False,
    )  # Parameter the prompt template
    return prompt
