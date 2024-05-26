// controllers/chatbot.js
const axios = require('axios');

exports.chatbotResponse = async (req, res) => {
    const userMessage = req.body.message;

    try {
        // Send the user's message to the chatbot API
        const response = await axios.post('http://your-chatbot-api-url', { message: userMessage });
        const botResponse = response.data;

        res.json(botResponse);
    } catch (error) {
        res.status(500).json({ error: error.message });
    }
};
