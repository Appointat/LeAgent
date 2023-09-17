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

from camel.agents.role_assignment_agent import RoleAssignmentAgent
from camel.configs import ChatGPTConfig
from camel.prompts import TextPrompt
from camel.societies import RolePlaying
from camel.typing import TaskType
from camel.utils import print_text_animated


def main(model_type=None) -> None:
    # task_prompt = "Develop a trading bot for the stock market."
    # task_prompt = "Simulate the session of learning Linear Regression between the tutor and student."
    task_prompt = "While teaching and learning, use linear regression in an e-commerce setting to determine the relationship between online advertising spend for a specific product and its subsequent sales on the platform?"

    model_config_description = ChatGPTConfig()
    role_assignment_agent = RoleAssignmentAgent(
        model=model_type, model_config=model_config_description)

    # Generate role with descriptions
    role_descriptions_instruction = "add one characteristic of tutor that he alsways ask questions to students to guide in their learning at the end of each round."
    role_description_dict = role_assignment_agent.run(task_prompt=task_prompt,
                                                      num_roles=2,
                                                      role_descriptions_instruction=
                                                        role_descriptions_instruction)
    for role_description in role_description_dict.values():
        role_description = "Remember you are in a teaching circumstance. " + role_description
    student_prompt = TextPrompt("""
[Student Configuration]
    - Depth: University
    - Learning-Style: Active
    - Communication-Style: Socratic
    - Emojis: Disabled (Default)
    - Language: English (Default)
    - Note: You can change the configuration by typing "/config" in the chatbox.
""")
    tutor_prompt = TextPrompt("""
[Tutor Configuration]
    - Depth: University, Prof
    - Teaching-Style: Reflective
    - Communication-Style: Socratic
    - Tone-Style: Encouraging
    - Reasoning-Framework: Causal
    - Emojis: Enabled (Default)
    - Language: English (Default)
                                 
[Overall Rules to follow]
    As a tutor, you should ask questions to the student to help them learn:
    1. Use emojis to make the content engaging
    2. Use bolded text to emphasize important points
    3. Do not compress your responses
    4. You can talk in any language

""")
    # role_description_dict.items()[0] += tutor_prompt
    # role_description_dict["AI Student"] = student_prompt
    # role_description_dict["AI Tutor in Machine Learning"] = tutor_prompt    

    # Split the original task into subtasks
    # subtasks = role_assignment_agent.split_tasks(task_prompt,
    #                                              role_description_dict)
    subtasks = [task_prompt]

    print(Fore.GREEN + 
          f"List of {len(role_description_dict)} roles with description:")
    for role_name in role_description_dict.keys():
        print(Fore.BLUE + f"{role_name}:\n"
              f"{role_description_dict[role_name]}\n")
    print(Fore.YELLOW + f"Original task prompt:\n{task_prompt}")
    print(Fore.YELLOW + f"List of {len(subtasks)} subtasks:")
    for i, subtask in enumerate(subtasks):
        print(Fore.YELLOW + f"Subtask {i + 1}: {subtask}")
    print(Fore.WHITE + "==========================================")
    
    for one_subtask in subtasks:
        role_compatibility_scores_dict = (
            role_assignment_agent.evaluate_role_compatibility(
                one_subtask, role_description_dict))
        
        # Get the top two roles with the highest compatibility scores
        top_two_positions = \
            sorted(role_compatibility_scores_dict.keys(),
                   key=lambda x: role_compatibility_scores_dict[x],
                   reverse=True)[:2]
        ai_assistant_role = top_two_positions[1]
        ai_user_role = top_two_positions[0] # The user role is the one with
                                            # the highest score/compatibility
        # ai_assistant_role = "AI Student"
        # ai_user_role = "AI Tutor in Machine Learning"
        ai_assistant_description = role_description_dict[ai_assistant_role]
        ai_user_description = role_description_dict[ai_user_role]

        print(Fore.WHITE + "==========================================")
        print(Fore.YELLOW + f"Subtask: \n{one_subtask}\n")
        print(Fore.GREEN + f"AI Assistant Role: {ai_assistant_role}\n"
              f"{ai_assistant_description}\n")
        print(Fore.BLUE + f"AI User Role: {ai_user_role}\n"
              f"{ai_user_description}\n")
        
        # You can use the following code to play the role-playing game
        sys_msg_meta_dicts = [
            dict(assistant_role=ai_assistant_role, user_role=ai_user_role,
                assistant_description=ai_assistant_description,
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

            input_assistant_msg = assistant_response.msg
        

if __name__ == "__main__":
    main()
