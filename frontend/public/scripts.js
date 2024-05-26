// // Functionality for contact form submission
// document.getElementById('contactForm').addEventListener('submit', function(event) {
//   event.preventDefault();
//   var name = document.getElementById('name').value;
//   var email = document.getElementById('email').value;
//   var message = document.getElementById('message').value;

//   // Simulated form submission
//   console.log('Name:', name);
//   console.log('Email:', email);
//   console.log('Message:', message);
  
//   alert('Your message has been sent!');
// // other webpages
// });

document.addEventListener("DOMContentLoaded", async function () {
  // Load components if needed

  // Handle Signup Form Submission
  if (document.getElementById("signupForm")) {
      document
          .getElementById("signupForm")
          .addEventListener("submit", async function (event) {
              event.preventDefault();
              const username = document.getElementById("signupUsername").value;
              const password = document.getElementById("signupPassword").value;

              try {
                  const response = await fetch("/api/user/signup", {
                      method: "POST",
                      headers: {
                          "Content-Type": "application/json",
                      },
                      body: JSON.stringify({ username, password }),
                  });

                  const result = await response.json();
                  if (response.ok) {
                      alert("Signup successful");
                  } else {
                      throw new Error(result.error || "Signup failed");
                  }
              } catch (error) {
                  console.error("Error:", error);
                  alert(error.message);
              }
          });
  }

  // Handle Signin Form Submission
  if (document.getElementById("signinForm")) {
      document
          .getElementById("signinForm")
          .addEventListener("submit", async function (event) {
              event.preventDefault();

              const username = document.getElementById("signinUsername").value;
              const password = document.getElementById("signinPassword").value;

              try {
                  const response = await fetch("/api/user/signin", {
                      method: "POST",
                      headers: {
                          "Content-Type": "application/json",
                      },
                      body: JSON.stringify({ username, password }),
                  });

                  const result = await response.json();

                  if (!response.ok) {
                      throw new Error(result.error || "Sign-in failed");
                  }

                  alert("Sign-in successful");
              } catch (error) {
                  console.error("Error:", error);
                  alert(error.message);
              }
          });
  }

  // Handle Chatbot Form Submission
  if (document.getElementById("chatbotForm")) {
      document
          .getElementById("chatbotForm")
          .addEventListener("submit", async function (event) {
              event.preventDefault();
              const userMessage = document.getElementById("userMessage").value;

              try {
                  const response = await fetch("/api/chatbot/message", {
                      method: "POST",
                      headers: {
                          "Content-Type": "application/json",
                      },
                      body: JSON.stringify({ message: userMessage }),
                  });

                  const result = await response.json();
                  if (response.ok) {
                      const chatbotMessages = document.getElementById("chatbot-messages");
                      const userMessageElement = document.createElement("div");
                      userMessageElement.classList.add("user-message");
                      userMessageElement.textContent = userMessage;
                      chatbotMessages.appendChild(userMessageElement);

                      const botMessageElement = document.createElement("div");
                      botMessageElement.classList.add("bot-message");
                      botMessageElement.textContent = result.response;
                      chatbotMessages.appendChild(botMessageElement);

                      document.getElementById("userMessage").value = "";
                  } else {
                      throw new Error(result.error || "Failed to get chatbot response");
                  }
              } catch (error) {
                  console.error("Error:", error);
                  alert(error.message);
              }
          });
  }

  // Functionality for contact form submission
  document.getElementById('contactForm').addEventListener('submit', function(event) {
      event.preventDefault();
      var name = document.getElementById('name').value;
      var email = document.getElementById('email').value;
      var message = document.getElementById('message').value;

      // Simulated form submission
      console.log('Name:', name);
      console.log('Email:', email);
      console.log('Message:', message);

      alert('Your message has been sent!');
      // other webpages
  });
});
