const GalleryItem = require('../models/galleryItem');

// Controller to get all gallery items
exports.getAllGalleryItems = async (req, res) => {
    try {
        const galleryItems = await GalleryItem.find();
        res.json(galleryItems);
    } catch (error) {
        res.status(500).json({ error: error.message });
    }
};
