const crypto = require('crypto');
const jwt = require('jsonwebtoken');

// Generate a random secret key
const generateRandomSecret = () => {
    return crypto.randomBytes(32).toString('hex');
};

// Use the generated secret key or set it as an environment variable
const secret = generateRandomSecret(); // Or retrieve it from process.env.JWT_SECRET

// Sign a JWT token
function signToken(payload) {
    return jwt.sign(payload, secret, { expiresIn: '1h' });
}

// Verify a JWT token
function verifyToken(token) {
    try {
        return jwt.verify(token, secret);
    } catch (error) {
        return null;
    }
}

module.exports = signToken;