// backend/middlewares/authenticationMiddleware.js

const jwt = require('jsonwebtoken');
const secret = '5430a173fe2274550a55e6b46acac59fd369ca755ee7a2895f78bdba9a8240d8'; // Use a separate secret for JWT tokens

const authenticate = (req, res, next) => {
    const token = req.header('Authorization').replace('Bearer ', '');
    if (!token) {
        return res.status(401).json({ message: 'No token, authorization denied' });
    }

    try {
        const decoded = jwt.verify(token, secret);
        req.user = decoded;
        next();
    } catch (err) {
        res.status(401).json({ message: 'Token is not valid' });
    }
};

module.exports = authenticate;  