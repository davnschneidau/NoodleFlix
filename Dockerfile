FROM node:14 as frontend-builder

WORKDIR /app/frontend

COPY frontend/package*.json ./

RUN npm install

COPY frontend/ .

RUN npm run build

FROM python:3.8 as backend-builder

WORKDIR /app/backend

COPY backend/requirements.txt ./

RUN pip install -r requirements.txt

COPY backend/ .

FROM node:14

WORKDIR /app

COPY --from=frontend-builder /app/frontend/build ./frontend/build

EXPOSE 3000
