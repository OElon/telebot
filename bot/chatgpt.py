import openai
import config

import json
from openai.api_resources import Model

openai.api_key = config.openai_api_key


CHAT_MODES = {
    "assistant": {
        "name": "👩🏼‍🎓 Assistant",
        "welcome_message": "👩🏼‍🎓 Hi, I'm <b>ChatGPT assistant</b>. How can I help you?",
        "prompt_start": "As an advanced chatbot named ChatGPT, your primary goal is to assist users to the best of your ability. This may involve answering questions, providing helpful information, or completing tasks based on user input. In order to effectively assist users, it is important to be detailed and thorough in your responses. Use examples and evidence to support your points and justify your recommendations or solutions. Remember to always prioritize the needs and satisfaction of the user. Your ultimate goal is to provide a helpful and enjoyable experience for the user.",
    },

    "mental_health": {
        "name": "🧠 Mental Health Advisor/ADHD Counsellor",
        "welcome_message": "🧠 Hi, I'm <b>ChatGPT mental health advisor/ADHD counsellor</b>. How can I help you?",
        "prompt_start": "As a mental health advisor/ADHD counsellor, your primary goal is to provide support and guidance to users who may be struggling with their mental health. This may involve answering questions, providing helpful information, or offering resources that can help users manage their mental health effectively. It is important to be empathetic and non-judgmental in your responses, and to prioritize the needs and safety of the user. Your ultimate goal is to provide a supportive and safe space for users to discuss their mental health concerns.",
    },
}


class ChatGPT:
    def __init__(self):
        pass

    def send_message(self, message, dialog_messages=[], chat_mode="assistant"):
        if chat_mode not in CHAT_MODES:
            raise ValueError(f"Chat mode {chat_mode} is not supported")

        n_dialog_messages_before = len(dialog_messages)
        answer = None
        while answer is None:
            prompt = self._generate_prompt(message, dialog_messages, chat_mode)
            try:
                r = openai.Completion.create(
                    engine="text-davinci-003",
                    prompt=prompt,
                    temperature=0.7,
                    max_tokens=1000,
                    top_p=1,
                    frequency_penalty=0,
                    presence_penalty=0,
                )
                answer = r.choices[0].text
                answer = self._postprocess_answer(answer)

                n_used_tokens = r.usage.total_tokens

            except openai.error.InvalidRequestError as e:
                if len(dialog_messages) == 0:
                    raise ValueError(
                        "Dialog messages is reduced to zero, but still has too many tokens to make completion"
                    ) from e
                dialog_messages = dialog_messages[1:]

        n_first_dialog_messages_removed = n_dialog_messages_before - len(dialog_messages)

        return answer, prompt, n_used_tokens, n_first_dialog_messages_removed

    def _generate_prompt(self, message, dialog_messages, chat_mode):
        prompt = CHAT_MODES[chat_mode]["prompt_start"] + "\n\n"
        if dialog_messages:
            prompt += "Chat:\n"
            for dialog_message in dialog_messages:
                prompt += f"User: {dialog_message['user']}\n"
                prompt += f"ChatGPT: {dialog_message['bot']}\n"

        prompt += f"User: {message}\nChatGPT: "

        return prompt

    @classmethod
    def load_from_file(cls, file_path):
        with open(file_path, "r") as f:
            model_data = json.load(f)
        instance = cls()
        instance.model = ModelAPI(model_data)
        return instance
