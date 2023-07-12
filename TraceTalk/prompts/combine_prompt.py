from langchain.prompts import PromptTemplate
from jinja2 import Template


# Combine prompt.
def combine_prompt(chat_history, query, answer_list, link_list):
    n = len(answer_list)

    template = f"""
Now I will provide you with {n} chains, here is the definition of chain: each chain contains an answer and a link. The answers in the chain are the results from the links.
In theory, each chain should produce a paragraph with links as references. It means that you MUST tell me from which references you make the summery.
The smaller the number of the chain, the more important the information contained in the chain.
Your final answer is verbose.
But if the meaning of an answer in a certain chain is similar to 'I am not sure about your question' or 'I refuse to answer such a question', it means that this answer chain is deprecated, and you should actively ignore the information in this answer chain.

You are not allowed to refuse to anwser the question.
You now are asked to try to answer and integrate these {n} chains (integration means avoiding repetition, writing logically, smooth writing, giving verbose answer), and divide it into 2-4 paragraphs appropriately.
The final answer is ALWAYS in the form of TEXT WITH MD LINK. If no refernce for one sentence, you do not need to attach the link to that sentence.
In addition, ALWAYS return "TEXT WITH MD LINK", and ALSO ALWAYs return a "REFERENCE" part in your answer (they are two parts).
ReFERENCE can ONLY be a list of links, each link is a reference for a sentence in the answer.

For exmaple:
I provide the input text:
CHAIN 1:
    CONTEXT:
        Text of chain 1. ABCDEFGHIJKLMNOPQRSTUVWXYZ
    REFERENCE:
        https://link1.com
CHAIN 2:
    CONTEXT: 
        Text of chain 2. ABCDEFGHIJKLMNOPQRSTUVWXYZ
    REFERENCE:
        https://link2.com
Your output should be:
COMBINATION:
    Text of combined chain 1 and chain 2. blablabla.
REFERENCE:
    [1] https://link1.com
    [2] https://link2.com
=========
"""
    template += """
QUESTION: {{query}}
Chat_history: 
{{chat_history}}[user]: {{query}}

=========
"""
    for i in range(n):
        template += f"""
### CHAIN {i+1}:
CONTEXT:
{answer_list[i]}
REFERENCE:
{link_list[i]}
"""
    template += """
=========
ANSWER THE QUESTION {{query}}, FINAL A VERBOSE ANSWER, language used for answers is CONSISTENT with QUESTION:
"""

    # prompt = PromptTemplate(template=template, input_variables=["query", "chat_history"], template_format="jinja2", validate_template=False) # Parameter the prompt template
    prompt = Template(template).render(query=query, chat_history=chat_history)
    return prompt
