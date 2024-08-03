class JSONTokenizer:
    def __init__(self):
        self.stack = []  # 用于保存当前的JSON片段，如开始一个对象或数组
        self.last_token_type = None

    def is_valid(self, current_state, token):
        # 检查token是否合法，并更新内部状态
        if self.is_start_of_object(token) and (
            self.last_token_type in [None, "start_array", "comma", "colon"]
        ):
            self.stack.append("object")
            self.last_token_type = "start_object"
            return True
        if self.is_start_of_array(token) and (
            self.last_token_type in [None, "start_array", "comma", "colon"]
        ):
            self.stack.append("array")
            self.last_token_type = "start_array"
            return True
        if self.is_end_of_object(token) and self.stack[-1] == "object":
            self.stack.pop()
            self.last_token_type = "end_object"
            return True
        if self.is_end_of_array(token) and self.stack[-1] == "array":
            self.stack.pop()
            self.last_token_type = "end_array"
            return True
        if self.is_key_or_value(token) and (
            self.last_token_type in ["start_object", "comma"]
        ):
            self.last_token_type = "key_or_value"
            return True
        if self.is_colon(token) and self.last_token_type == "key_or_value":
            self.last_token_type = "colon"
            return True
        if self.is_comma(token) and self.last_token_type in [
            "key_or_value",
            "end_object",
            "end_array",
        ]:
            self.last_token_type = "comma"
            return True
        return False
