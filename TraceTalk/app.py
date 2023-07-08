import sys
import os
import codecs

core_directory = os.path.dirname(os.path.abspath(__file__))
if core_directory not in sys.path:
    sys.path.append(core_directory)  # add the core directory to the path
from main import main as agent

from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)


sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())


@app.route("/chatbot_agent", methods=["POST"])
def handler():
    print("Received request...")
    # return jsonify({'answer': 'Hello World!'})
    try:
        data = request.get_json()
        messages = data.get("messages")  # Get 'message' field from the JSON object
        print("Received messages:")
        for messgae in messages:
            print(messgae)

        messages_str_list = [message.get("content") for message in messages]

        # send the message to OpenAI API
        answer = agent(messages=messages_str_list)

        return jsonify({"answer": answer})
    except Exception as e:
        print(f"Error occurred: {e}")
        return jsonify({"error": str(e)})


if __name__ == "__main__":
    print("Starting the server...")
    app.run(debug=True, host="0.0.0.0", port=5020)
