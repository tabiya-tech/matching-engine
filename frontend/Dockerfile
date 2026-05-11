FROM node:20-alpine AS build
WORKDIR /app
COPY package*.json ./
RUN npm ci
COPY . .
RUN mkdir -p public/data && [ -f public/data/supply.jsonl ] || touch public/data/supply.jsonl
ARG VITE_API_BASE
ARG VITE_API_KEY
ENV VITE_API_BASE=${VITE_API_BASE}
ENV VITE_API_KEY=${VITE_API_KEY}
RUN npm run build

FROM node:20-alpine AS runtime
WORKDIR /app
RUN npm i -g serve
COPY --from=build /app/dist ./dist
ENV PORT=8080
CMD ["sh", "-c", "serve -s dist -l ${PORT}"]
