const mongoose = require('mongoose');

// Define the schema for gallery items
const galleryItemSchema = new mongoose.Schema({
    title: { type: String, required: true },
    description: { type: String, required: true },
    imageUrl: { type: String, required: true },
    createdAt: { type: Date, default: Date.now }
});

// Create the model from the schema
const GalleryItem = mongoose.model('GalleryItem', galleryItemSchema);

module.exports = GalleryItem;
