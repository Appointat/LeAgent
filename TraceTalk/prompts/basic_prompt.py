from langchain.prompts import PromptTemplate


# Prompt the chatbot.
def basic_prompt():
    template = """
In your answer you should add a part called RESOURCE to extract the corresponding links from CONTEXT and list them in RESOURCE in markdown and citation format.
Strictly PROHIBITED to create or fabricate the links within RESOURCE, if no links are found please say sorry. The RESOURCE should ONLY consist of LINKS that are directly drawn from the CONTEXT.
If the answer to the QUESTION is not within your knowledge scope, admit it instead of concocting an answer. 
In the event where the QUESTION doesn't correlate with the CONTEXT, it's acceptable to respond with an apology, indicating that more information is required for an accurate answer, or you may respectfully decline to provide an answer.

===== CONTEXT =====
{{context}}

===== CHAT HISTORY =====
{{chat_history}}

===== RESOURCE =====
{{resource}}

=========
ANSWER THE QUESTION "{{query}}", FINAL A VERBOSE ANSWER, language used for answers is CONSISTENT with QUESTION:
"""
    prompt = PromptTemplate(
        template=template,
        input_variables=["context", "chat_history", "resource","query"],
        template_format="jinja2",
        validate_template=False,
    )  # Parameter the prompt template
    return prompt
