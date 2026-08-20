# Technical Test 

## Objective 

Create the backend component of a research agent to demonstrate your technical skills and experience in machine learning integration. The service should work with the provided UI, allowing users to ask the agent research queries, and streaming back the answers. Users should be able to upload source files for embedding that the agent may use in addition to external sources.

## Requirements 

### Frontend 

Provided is a React frontend that: 

* Provides a text input for the user to make a research request 
* Streams in the markdown response as it is generated 
* Provides an uploader for users to store their own information sources


### Backend 

Develop an API to handle requests from the frontend. You may use any libraries you see fit. Errors should be handled/reported properly and parallelism should be used where appropriate. 

* Receive rest requests from the provided frontend 
* Stream responses back to the provided frontend
* Embed uploaded source files & query at the correct times 
* Build an LLM research agent that acts on the user’s research request 
* The agent should be able to utilise external sources


## Submission Requirements 

### Codebase: Build & provide a link to the repo(s) created for your work. 

### Documentation

1. Instructions to set up and run the project. 
2. Provide a brief explanation of the architecture and design decisions. Include any plans you had that you may not have had time for. 
3. Example queries and expected outputs. 
4. How and why you would evaluate your agent so that it could be improved on.

## Bonus Points
If you have time and feel up to an extra challenge, pick and choose any of the following items to include in your submission: 
- Use Docker to containerize the application.
- Add unit tests for key components.
- Queuing for file uploads.