const axios = require('axios');


const data = {
    message: "Hello",
};

axios.post('http://localhost:5000/chatbot_agent', data)
.then(function (response) {
    console.log(response.data);
})
.catch(function (error) {
    console.log(error);
});