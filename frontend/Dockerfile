FROM node:22-alpine

WORKDIR /app
COPY package.json package-lock.json ./
RUN npm config set registry https://registry.npmmirror.com && \
    npm ci

COPY . .
EXPOSE 5173
