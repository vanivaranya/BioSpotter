const express = require('express');
const router = express.Router();
const { getAllDiscussions, createDiscussion } = require('../controllers/discussion');

// Define routes for discussions
router.get('/', getAllDiscussions);
router.post('/', createDiscussion);

module.exports = router;
    