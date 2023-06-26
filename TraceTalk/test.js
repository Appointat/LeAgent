const axios = require('axios');



const Role = {
    ASSISTANT: 'assistant',
    USER: 'user'
};


function Message(role, content) {
    this.role = role;
    this.content = content;
}
messages = []
message1 = new Message(Role.USER, 'Hello.');
message2 = new Message(Role.ASSISTANT, 'Hi.');
messages.push(message1);
messages.push(message2);


const data = {
    messages: messages,
};


axios.post('http://localhost:5020/chatbot_agent', data)
.then(function (response) {
    console.log(response.data);
})
.catch(function (error) {
    console.log(error);
});