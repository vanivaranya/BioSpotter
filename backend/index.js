const express = require('express');
const cors = require('cors');
const bodyParser = require('body-parser');
const path = require('path');
const multer = require('multer');
const { signIn, signUp } = require('./controllers/user');
const userRoutes = require('./routes/user');
const discussionRoutes = require('./routes/discussionRoutes');
const galleryRoutes = require('./routes/galleryRoutes');
const logReqRes = require('./middlewares/requestResponseLogger');
const connectMongoDb = require('./connection');
const axios = require('axios');
const fs = require('fs');
const mongoose = require('mongoose');
const http = require('http');
const socketIo = require('socket.io');

const app = express();
const PORT = process.env.PORT || 3001;

// Middleware setup
app.use(cors());
app.use(bodyParser.json());

// Serve static files from frontend public directory
app.use(express.static(path.join(__dirname, '..', 'frontend', 'public')));

// Configure multer for file uploads
const upload = multer({ dest: path.join(__dirname, '..', 'uploads') });
app.use(upload.single('photo'));

// Connect to MongoDB
const MONGO_URI = 'mongodb://127.0.0.1:27017/animal-heritage-db';
connectMongoDb(MONGO_URI);

// Logging middleware
const requestResponseLogger = logReqRes('request_response.log');
app.use(requestResponseLogger);

// API routes
app.post("/api/signin", signIn);
app.post("/api/signup", signUp);
app.use("/api/user", userRoutes);
app.use("/api/discussions", discussionRoutes);
app.use("/api/gallery", galleryRoutes);

// Prediction route
app.post('/predict', async (req, res) => {
  try {
    // Handle prediction logic
    const file = req.file;

    if (!file) {
      throw new Error('No file uploaded');
    }

    const filePath = path.join(__dirname, '..', 'uploads', file.filename);

    const formData = new FormData();
    formData.append('file', fs.createReadStream(filePath));

    const flaskResponse = await axios.post('http://localhost:5000/predict', formData, {
      headers: {
        'Content-Type': 'multipart/form-data'
      }
    });

    res.json(flaskResponse.data);
  } catch (error) {
    console.error(error.message);
    res.status(500).json({ error: 'Failed to process prediction request' });
  }
});

// Error handling middleware
app.use((err, req, res, next) => {
  console.error(err.stack);
  res.status(500).json({ error: err.message });
});

// Start server and socket.io connection
const server = http.createServer(app);
const io = socketIo(server);

io.on('connection', (socket) => {
  console.log('A user connected');

  socket.on('post', (postContent) => {
    io.emit('new post', postContent);
  });

  socket.on('disconnect', () => {
    console.log('A user disconnected');
  });
});

server.listen(PORT, () => {
  console.log(`Server started on http://localhost:${PORT}`);
});