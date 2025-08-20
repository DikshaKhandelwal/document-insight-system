# -----------------------------
# Stage 1 - Build Next.js App
# -----------------------------
FROM node:20-slim AS frontend-builder

WORKDIR /app

# Install deps separately for caching
COPY package.json pnpm-lock.yaml tsconfig.json next.config.mjs next-env.d.ts postcss.config.mjs tailwind.config.js ./
RUN npm config set registry https://registry.npmmirror.com
RUN corepack enable && pnpm install --frozen-lockfile
RUN npm install -g pm2

# Copy rest and build
COPY app/ ./app/
COPY components/ ./components/
COPY public/ ./public/
COPY styles/ ./styles/
COPY lib/ ./lib/
RUN npm run build

# -----------------------------
# Stage 2 - Final Runtime Image
# -----------------------------
FROM python:3.11-slim AS runtime

WORKDIR /app

# Install system deps (only what’s needed)
RUN apt-get update && apt-get install -y \
 curl nginx nodejs \
 && rm -rf /var/lib/apt/lists/*

# Copy pm2 from frontend-builder stage
COPY --from=frontend-builder /usr/local/lib/node_modules/pm2 /usr/local/lib/node_modules/pm2
COPY --from=frontend-builder /usr/local/bin/pm2 /usr/local/bin/pm2
COPY --from=frontend-builder /usr/local/bin/pm2-runtime /usr/local/bin/pm2-runtime

# -----------------------------
# Python deps
# -----------------------------
COPY backend/requirements.txt /app/backend/requirements.txt
RUN pip install --no-cache-dir -r /app/backend/requirements.txt

# -----------------------------
# Copy backend + frontend
# -----------------------------
COPY backend /app/backend
COPY --from=frontend-builder /app/.next /app/.next
COPY --from=frontend-builder /app/node_modules /app/node_modules
COPY --from=frontend-builder /app/public /app/public
COPY --from=frontend-builder /app/package.json /app/package.json


# -----------------------------
# Nginx config
# -----------------------------
COPY nginx.conf /etc/nginx/nginx.conf

# -----------------------------
# PM2 config
# -----------------------------
COPY pm2.json /app/pm2.json

EXPOSE 8080

CMD ["pm2-runtime", "start", "pm2.json"]
