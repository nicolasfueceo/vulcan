import autogen
import pandas as pd


class ResearchAgent:
    def __init__(self, llm_config: dict):
        self.llm_config = llm_config
        self.analyst = autogen.AssistantAgent(
            name="Data_Analyst",
            system_message="You are a senior data analyst. Your role is to perform detailed data analysis using Python. Given a task, write Python code to analyze the provided pandas DataFrame. Your code should be executable in a Jupyter environment. Do not offer analysis or interpretations, only code. IMPORTANT: After your code block, on a new line, write the word TERMINATE to end your turn. Do not write TERMINATE inside the code block.",
            llm_config=self.llm_config,
        )
        self.scientist = autogen.AssistantAgent(
            name="Data_Scientist",
            system_message="You are a data scientist. Your role is to interpret the results of data analysis, generate insights, and propose new hypotheses. You will be given the results of a data analysis task and are expected to provide a summary of insights and suggest next steps. When you are done, on a new line, write the word TERMINATE to end your turn.",
            llm_config=self.llm_config,
        )
        self.user_proxy = autogen.UserProxyAgent(
            name="User_Proxy",
            human_input_mode="NEVER",
            max_consecutive_auto_reply=5,
            code_execution_config={
                "work_dir": "runtime/generated_code",
                "use_docker": False,
            },
        )

    def _custom_speaker_selection_func(
        self, last_speaker: autogen.Agent, groupchat: autogen.GroupChat
    ) -> autogen.Agent:
        """Custom speaker selection logic."""
        messages = groupchat.messages
        if len(messages) <= 1:
            return self.analyst

        # If the last speaker was the analyst, the user_proxy should execute the code
        if last_speaker.name == "Data_Analyst":
            return self.user_proxy

        # If the last speaker was the user_proxy (after execution), the scientist should speak
        if last_speaker.name == "User_Proxy":
            return self.scientist

        # Otherwise, the analyst should speak
        return self.analyst

    def run_eda_task(self, df_reviews: pd.DataFrame, df_items: pd.DataFrame, task: str):
        groupchat = autogen.GroupChat(
            agents=[self.user_proxy, self.analyst, self.scientist],
            messages=[],
            speaker_selection_method=self._custom_speaker_selection_func,
        )
        manager = autogen.GroupChatManager(
            groupchat=groupchat, llm_config=self.llm_config
        )

        # Build the initial message, including dataframes if they exist
        message = f"Perform the following task: {task}.\n"
        if df_reviews is not None:
            message += f"""
            Here is the df_reviews dataframe:
            {df_reviews.head().to_string()}
            """
        if df_items is not None:
            message += f"""
            Here is the df_items dataframe:
            {df_items.head().to_string()}
            """
        message += """
        First, the analyst will write the code.
        Then, the user_proxy will execute it.
        Finally, the scientist will interpret the results.
        """

        # The user proxy will execute the code written by the analyst
        self.user_proxy.initiate_chat(
            manager,
            message=message,
        )

        # The result of the chat will be in the chat history.
        return groupchat.messages
