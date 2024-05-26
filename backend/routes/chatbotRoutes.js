// routes/chatbotRoutes.js
const express = require('express');
const router = express.Router();
const chatbotController = require('../controllers/chatbot');

router.post('/message', chatbotController.chatbotResponse);

module.exports = router;
