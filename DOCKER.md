# Docker Quick Start Guide

Run all 3 MCP services with Docker Compose in one command!

## Prerequisites

- Docker Desktop installed
- Docker Compose included (comes with Docker Desktop)

## Quick Start

### Start All Services

```bash
# Build and start all services
docker-compose up --build

# Or run in detached mode (background)
docker-compose up -d --build
```

This starts:
- **plot-service** on `http://localhost:8000`
- **calculator** on `http://localhost:8001`
- **pdf-parser** on `http://localhost:8002`

### Stop All Services

```bash
# Stop services (keeps containers)
docker-compose stop

# Stop and remove containers
docker-compose down

# Stop and remove containers + volumes
docker-compose down -v
```

### View Logs

```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f plot-service
```

### Check Service Status

```bash
docker-compose ps
```

## Using with Streamlit

Once services are running, configure Streamlit to use localhost endpoints:

```bash
export PLOT_SERVICE_URL=http://localhost:8000/mcp/plot
export CALCULATOR_URL=http://localhost:8001/mcp/calculate
export PDF_PARSER_URL=http://localhost:8002/mcp/parse

# Start Streamlit
streamlit run frontend/app.py
```

## Troubleshooting

### Port Already in Use

If you see port conflicts:

```bash
# Check what's using the ports
netstat -ano | findstr :8000
netstat -ano | findstr :8001
netstat -ano | findstr :8002

# Change ports in docker-compose.yml:
# ports:
#   - "9000:8000"  # Change 8000 to 9000
```

### Rebuild After Code Changes

```bash
# Rebuild specific service
docker-compose build plot-service

# Rebuild all services
docker-compose build
```

### Clean Start

```bash
# Remove all containers and rebuild
docker-compose down
docker-compose up --build
```

## Advantages Over Manual Startup

✅ **One command** to start all services
✅ **Automatic restart** if a service crashes
✅ **Health checks** to verify services are running
✅ **Network isolation** with dedicated mcp-network
✅ **Easy cleanup** with `docker-compose down`
✅ **Consistent environment** across different machines

## Production Deployment

For production, use Cloud Run (already deployed):
- plot-service: https://plot-service-347876502362.us-central1.run.app
- calculator: https://calculator-h7whjphxza-uc.a.run.app
- pdf-parser: https://pdf-parser-h7whjphxza-uc.a.run.app

See [Deployment Guide](docs/deployment.md) for details.
