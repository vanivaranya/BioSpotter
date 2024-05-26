const express = require('express');
const galleryController = require('../controllers/gallery');
const router = express.Router();

// Define route to get all gallery items
router.get('/', galleryController.getAllGalleryItems);

module.exports = router;
