const fs = require("fs");

function logReqRes(filename) {
  return (req, res, next) => {
    const logMessage = `${new Date().toISOString()}: ${req.method} ${req.url}\n` +
      `   Params: ${JSON.stringify(req.params)}\n` +
      `   Query: ${JSON.stringify(req.query)}\n` +
      `   Headers: ${JSON.stringify(req.headers)}\n` +
      `   Body: ${JSON.stringify(req.body)}\n`; // Log request body

    fs.appendFile(filename, logMessage, (err) => {
      if (err) {
        console.error("Error logging request:", err);
      }
      next(); // Continue request processing regardless of error
    });

    // Capture the response data before sending it
    const originalSend = res.send;
    res.send = function (body) {
      // Log the response body
      const responseLog = `\nResponse Body: ${JSON.stringify(body)}\n`;
      fs.appendFile(filename, responseLog, (err) => {
        if (err) {
          console.error("Error logging response:", err);
        }
      });

      // Call the original res.send() function to send the response
      originalSend.call(this, body);
    };
  };
}

module.exports = logReqRes;
