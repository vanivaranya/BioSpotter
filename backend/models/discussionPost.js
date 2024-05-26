const mongoose = require('mongoose');
const Schema = mongoose.Schema;

const DiscussionPostSchema = new Schema({
    username: { type: String, required: true },
    content: { type: String, required: true },
    createdAt: { type: Date, default: Date.now }
});

const DiscussionPost = mongoose.model('DiscussionPost', DiscussionPostSchema);

module.exports = DiscussionPost;
