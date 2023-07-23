from langchain.prompts import PromptTemplate


# Prompt the chatbot.
def basic_prompt():
    template = """
Given the following CONTEXT part of a long document and a QUESTION, create an answer with RESOURCE.
If you don't know the answer, just say that you don't know. Don't try to make up an answer. 
If the QUESTION is not associated with the CONTEXT, you can say "SORRY" and say that you need more information to answer it, or you can enven refuse to answer it.
ALWAYS return a "RESOURCE" part in your answer. RESOURCE can ONLY be a list of LINK, and is located in the end of your answer.

=========
CONTEXT
{{context}}

=========
Chat_history:
{{chat_history}}
=========
FINAL ANSWER THE QUESTION "{{query}}", answer language is CONSISTENT with QUESTION:
"""
    prompt = PromptTemplate(
        template=template,
        input_variables=["context", "chat_history", "query"],
        template_format="jinja2",
        validate_template=False,
    )  # Parameter the prompt template
    return prompt
