# PhoBert Medical Chatbot

## Setup front-end
1. npm install
2. npm run dev

## Setup back-end
1. JDK 21
2. Intellij
3. Get .env file from me
4. Mysql: -> root, pass-> 123456

## Setup fast-api
1. Get .env file from me


## Update submodules repo
### Structure
chatbot_app/          <- repo chính
├── backend_chatbot/  <- submodule

1. Update submodule
cd backend_chatbot
git add .
git commit -m "Cập nhật logic chatbot"
git push origin main

2. Update main repo
cd ..
git add backend_chatbot
git commit -m "Update backend_chatbot submodule to latest commit"
git push origin web-app



