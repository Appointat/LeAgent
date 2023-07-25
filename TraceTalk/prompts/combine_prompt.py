import re
from langchain.prompts import PromptTemplate
from jinja2 import Template
from src import get_tokens_number


# Combine prompt.
def combine_prompt(chat_history, query, answer_list, link_list_list):
    n = len(answer_list)

    template = f"""
===== RULES =====
Now I will provide you with {n} chains, here is the definition of chain: each chain contains an answer and a link. The answers in the chain are the results from the links.
In theory, each chain should produce a paragraph with links as the resources. It means that you MUST tell me from which references you make the summery.
The smaller the number of the chain, the more important the information contained in the chain.
Your final answer is verbose.
But if the meaning of an answer in a certain chain is similar to 'I am not sure about your question' or 'I refuse to answer such a question', it means that this answer chain is deprecated, and you should actively ignore the information in this answer chain.

You now are asked to try to answer and integrate these {n} chains (integration means avoiding repetition, writing logically, smooth writing, giving verbose answer), and answer it in 2-4 paragraphs appropriately.
The final answer is ALWAYS in Markdown format.
Provide your answer in a style of CITATION format where you also list the resources from where you found the information at the end of the text. (an example is provided below)
In addition, in order to demostrate the knowledge resources you have referred, please ALWAYs return a "RESURCE" part in your answer. 
RESOURCE can ONLY be a list of links, and each link means the knowledge resource of each chain. Each chain has only one RESOURCE part.

===== EXAMPLE =====
For exmaple, if you are provided with 2 chains, the template is below:
CHAIN 1:
    CONTEXT:
        Text of chain 1. ABCDEFGHIJKLMNOPQRSTUVWXYZ
    RESOURCE:
        https://link1.com
CHAIN 2:
    CONTEXT: 
        Text of chain 2. ABCDEFGHIJKLMNOPQRSTUVWXYZ
    RESOURCE:
        https://link2.com

YOU COMPLETE ANSWER LIKE THIS:
    Integrated text of chain 1 [1] and chain 2 [2]. Blablabla.
REFERENCE:
    [1] [title_link1](https://link1.com)
    [2] [title_link2](https://link2.com)

"""
    
    chat_history_text ="""
===== CHAT HISTORY =====
{{chat_history}}

"""
    template += chat_history_text

    init_chain_tmp = f"Now I provide you with {n} chains:"
    template += init_chain_tmp
    for i in range(n):
        link_list = '\n'.join([item for item in link_list_list[i]])
        template_tmp = f"""
===== CHAIN {i+1} =====
CONTEXT:
{answer_list[i]}
RESOURCE:
{link_list}
"""
        length_prompt = len(re.findall(r"\b\w+\b", template + template_tmp))
        if get_tokens_number(template + template_tmp) > 3800:
            break
        template += template_tmp
    # After breaking from the loop, print the remaining links.
    for j in range(i + 1, n):
        link_list = '\n'.join([item for item in link_list_list[j]])
        template_tmp = f"{link_list}\n"
        if get_tokens_number(template + template_tmp) > 3800:
            break
        template += template_tmp

    template += """
=========
ANSWER THE QUESTION "{{query}}", FINAL A VERBOSE ANSWER, language used for answers is CONSISTENT with QUESTION:
"""

    # prompt = PromptTemplate(template=template, input_variables=["query", "chat_history"], template_format="jinja2", validate_template=False) # Parameter the prompt template
    prompt = Template(template).render(query=query, chat_history=chat_history)
    return prompt
