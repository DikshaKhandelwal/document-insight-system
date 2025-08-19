# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set environment variables for Python
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set workdir to /app
WORKDIR /app

# Copy backend code
COPY backend/ ./backend/

# Copy lib (for PDF extractors)
COPY lib/ ./lib/

# Copy requirements
COPY backend/requirements.txt ./backend/requirements.txt


# --- Install system dependencies for PyMuPDF and other libs ---
RUN apt-get update && \
 apt-get install -y ca-certificates build-essential libglib2.0-0 libsm6 libxext6 libxrender-dev curl && \
 pip install --upgrade pip && \
 pip install 'numpy<2' && \
 pip install -r backend/requirements.txt

# --- Install Node.js, pnpm, and build the Next.js frontend ---
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - && \
 apt-get install -y nodejs && \
 npm install -g pnpm

# Copy frontend files
COPY package.json ./
COPY pnpm-lock.yaml ./
COPY tsconfig.json ./
COPY next.config.mjs ./
COPY next-env.d.ts ./
COPY postcss.config.mjs ./
COPY tailwind.config.js ./
COPY app/ ./app/
COPY components/ ./components/
COPY public/ ./public/
COPY styles/ ./styles/


# Install frontend dependencies and build
RUN pnpm install --frozen-lockfile && pnpm build

# Copy the built frontend static files to /app/build for backend serving
RUN mkdir -p /app/build && cp -r .next /app/build/ && cp -r public /app/build/

# Expose port (FastAPI default is 8000, but map to 8080 in docker run)
EXPOSE 8000

# Set environment for FastAPI to find lib extractors
ENV PYTHONPATH="/app/lib/pdf-extractors:/app/backend:/app/lib"

# Entrypoint: run FastAPI app
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8080"]
