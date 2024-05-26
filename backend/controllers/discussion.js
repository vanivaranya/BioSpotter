const DiscussionPost = require('../models/discussionPost');

exports.getAllDiscussions = async (req, res) => {
    try {
        const discussions = await DiscussionPost.find().sort({ createdAt: -1 });
        res.json(discussions);
    } catch (error) {
        res.status(500).json({ error: error.message });
    }
};

exports.createDiscussion = async (req, res) => {
    try {
        const { username, content } = req.body;
        const newDiscussion = new DiscussionPost({ username, content });
        await newDiscussion.save();
        res.status(201).json(newDiscussion);
    } catch (error) {
        console.error(error);
        res.status(500).json({ error: "Failed to create discussion post" });
    }
};