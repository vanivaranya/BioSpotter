const mongoose = require('mongoose');

function connectMongoDb(url) {
    if (!url) {
        throw new Error('MongoDB URL is not provided');
    }

    mongoose.connect(url);

    const connection = mongoose.connection;
    connection.once('open', () => {
        console.log('MongoDB connection established successfully');
    });
    connection.on('error', (err) => {
        console.error('MongoDB connection error:', err);
    });
}

module.exports = connectMongoDb;

// const mongoose = require("mongoose");

// async function connectMongoDb(url) {
//   try {
//     await mongoose.connect(url, {
//       useNewUrlParser: true,
//       useUnifiedTopology: true,
//     });
//     console.log("MongoDB connected");
//   } catch (error) {
//     console.error("Error connecting to MongoDB:", error);
//     throw error;
//   }
// }

// module.exports = {
//   mongoose,
//   connectMongoDb
// };
