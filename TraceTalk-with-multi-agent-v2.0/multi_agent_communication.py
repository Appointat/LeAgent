# =========== Copyright 2023 @ CAMEL-AI.org. All Rights Reserved. ===========
# Licensed under the Apache License, Version 2.0 (the “License”);
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an “AS IS” BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========== Copyright 2023 @ CAMEL-AI.org. All Rights Reserved. ===========
from colorama import Fore
import json

from camel.agents.insight_agent import InsightAgent
from camel.agents.role_assignment_agent import RoleAssignmentAgent
from camel.configs import ChatGPTConfig
from camel.societies import RolePlaying
from camel.typing import TaskType
from camel.utils import print_text_animated


def main(model_type=None) -> None:
    task_prompt = "Develop a trading bot for the stock market."
    # task_prompt = "Show me python code implementing the deep first traverse."
    # task_prompt = "Write a easy blog about Computer Sci   ence Education."

    model_config_description = ChatGPTConfig()
    role_assignment_agent = RoleAssignmentAgent(
        model=model_type, model_config=model_config_description)
    insight_agent = InsightAgent(model=model_type,
                                 model_config=model_config_description)

    # Generate role with descriptions
    role_descriptions_dict = role_assignment_agent.run(task_prompt=task_prompt,
                                                      num_roles=3)

    # Split the original task into subtasks
    subtasks_with_dependencies_dict = \
        role_assignment_agent.split_tasks(task_prompt=task_prompt,
                                          role_descriptions_dict=\
                                            role_descriptions_dict,
                                          context_text=task_prompt)
    print(Fore.BLUE + "Dependencies among subtasks: " +
          json.dumps(subtasks_with_dependencies_dict, indent=4))
    subtasks = [
        subtasks_with_dependencies_dict[key]["description"]
        for key in sorted(subtasks_with_dependencies_dict.keys())
    ]

    parallel_subtask_pipelines = \
        role_assignment_agent.get_task_execution_order(
            subtasks_with_dependencies_dict)

    # Record the insights from chat history of the assistant
    insights_pre_subtasks = {ID_subtask: "" for ID_subtask
                             in subtasks_with_dependencies_dict.keys()}

    print(Fore.GREEN + 
          f"List of {len(role_descriptions_dict)} roles with description:")
    for role_name in role_descriptions_dict.keys():
        print(Fore.BLUE + f"{role_name}:\n"
              f"{role_descriptions_dict[role_name]}\n")
    print(Fore.YELLOW + f"Original task prompt:\n{task_prompt}")
    print(Fore.YELLOW + f"List of {len(subtasks)} subtasks:")
    for i, subtask in enumerate(subtasks):
        print(Fore.YELLOW + f"Subtask {i + 1}:\n{subtask}")
    for idx, subtask_group in enumerate(parallel_subtask_pipelines, 1):
        print(Fore.YELLOW + f"Pipeline {idx}: {', '.join(subtask_group)}")
    print(Fore.WHITE + "==========================================")

    # Resolve the subtasks in sequence based on the dependency graph
    for ID_one_subtask in (subtask for pipeline in parallel_subtask_pipelines
                           for subtask in pipeline):
        # Get the description of the subtask
        one_subtask = \
            subtasks_with_dependencies_dict[ID_one_subtask]["description"]
        # Get the insights from the chat history of based on the dependencies
        ID_pre_subtasks = \
            subtasks_with_dependencies_dict[ID_one_subtask]["dependencies"]

        if len(ID_pre_subtasks) != 0:
            insights_pre_subtask = \
                "====== NovaDive & QuestXplorer of PREVIOUS CONVERSATION " + \
                "ROUND =====\n" + \
                "NovaDive and QuestXplorer are agent names we " +\
                "brainstormed for a system designed to decompose text or " + \
                "code, identify post-2022 unknowns, and craft insightful " + \
                "questions based on prior conversation rules. \n" + \
                "The achievements or insights of previous conversation " + \
                "are following:\n" + \
                    "\n\n".join(insights_pre_subtasks[pre_subtask]
                                for pre_subtask in ID_pre_subtasks)
        else:
            insights_pre_subtask = ""
        print(Fore.WHITE + insights_pre_subtask + "\n")

        # Get the role with the highest compatibility score
        role_compatibility_scores_dict = (
            role_assignment_agent.evaluate_role_compatibility(
                one_subtask, role_descriptions_dict))

        # Get the top two roles with the highest compatibility scores
        top_two_positions = \
            sorted(role_compatibility_scores_dict.keys(),
                   key=lambda x: role_compatibility_scores_dict[x],
                   reverse=True)[:2]
        ai_assistant_role = top_two_positions[1]
        ai_user_role = top_two_positions[0] # The user role is the one with
                                            # the highest score/compatibility
        ai_assistant_description = role_descriptions_dict[ai_assistant_role]
        ai_user_description = role_descriptions_dict[ai_user_role]

        print(Fore.WHITE + "==========================================")
        print(Fore.YELLOW + f"Subtask: \n{one_subtask}\n")
        print(Fore.GREEN + f"AI Assistant Role: {ai_assistant_role}\n"
              f"{ai_assistant_description}\n")
        print(Fore.BLUE + f"AI User Role: {ai_user_role}\n"
              f"{ai_user_description}\n")

        # You can use the following code to play the role-playing game
        sys_msg_meta_dicts = [
            dict(assistant_role=ai_assistant_role, user_role=ai_user_role,
                assistant_description=ai_assistant_description + \
                    insights_pre_subtask,
                user_description=ai_user_description) for _ in range(2)
        ]

        role_play_session = RolePlaying(
            assistant_role_name=ai_assistant_role,
            user_role_name=ai_user_role,
            task_prompt=one_subtask,
            model_type=model_type,
            task_type=TaskType.ROLE_DESCRIPTION,  # Important for role description
            with_task_specify=False,
            task_specify_agent_kwargs=dict(model=model_type),
            extend_sys_msg_meta_dicts=sys_msg_meta_dicts,
            output_language="zh"
        )

        chat_history_assistant = f"The TASK of the context text is:\n{one_subtask}\n"

        chat_turn_limit, n = 50, 0
        input_assistant_msg, _ = role_play_session.init_chat()
        while n < chat_turn_limit:
            n += 1
            assistant_response, user_response = role_play_session.step(
                input_assistant_msg)

            if assistant_response.terminated:
                print(Fore.GREEN + (
                    f"{ai_assistant_role} terminated. "
                    f"Reason: {assistant_response.info['termination_reasons']}."))
                break
            if user_response.terminated:
                print(Fore.GREEN + (
                    f"{ai_user_role} terminated. "
                    f"Reason: {user_response.info['termination_reasons']}."))
                break

            print_text_animated(
                Fore.BLUE +
                f"AI User: {ai_user_role}\n\n{user_response.msg.content}\n")
            print_text_animated(Fore.GREEN +
                                f"AI Assistant: {ai_assistant_role}\n\n" +
                                f"{assistant_response.msg.content}\n")

            if "CAMEL_TASK_DONE" in user_response.msg.content:
                break

            # Generate the insights from the chat history
            chat_history_assistant += (f"===== [{n}] ===== \n"
                                       f"{user_response.msg.content}\n"
                                       f"{assistant_response.msg.content}\n")

            input_assistant_msg = assistant_response.msg

        insights_instruction = ("The CONTEXT TEXT is related to code implementation. " +
                                "Pay attention to the code structure code environment.")
        insights = insight_agent.run(context_text=chat_history_assistant)
        insights_str = insight_agent.convert_json_to_str(insights)
        insights_pre_subtasks[ID_one_subtask] = insights_str
        

if __name__ == "__main__":
    main()
