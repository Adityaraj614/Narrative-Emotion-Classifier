# src/conversation_memory.py

class ConversationMemory:
    def __init__(self, max_history=10):
        self.history = []
        self.max_history = max_history

    def add_message(self, message):
        self.history.append(message)

        # keep only last N messages
        if len(self.history) > self.max_history:
            self.history.pop(0)

    def get_history(self):
        return self.history

    def clear(self):
        self.history = []