#!/bin/bash

# VULCAN Production Launcher using PM2
# Provides better process management and monitoring

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_status() {
    echo -e "${BLUE}[VULCAN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

# Check if PM2 is installed
if ! command -v pm2 >/dev/null 2>&1; then
    print_error "PM2 is not installed. Installing PM2 globally..."
    npm install -g pm2 || {
        print_error "Failed to install PM2. Please run: npm install -g pm2"
        exit 1
    }
fi

# Create PM2 ecosystem file
print_status "Creating PM2 ecosystem configuration..."
cat > ecosystem.config.js << 'EOF'
module.exports = {
  apps: [
    {
      name: 'vulcan-api',
      cwd: './src',
      script: 'python',
      args: '-m uvicorn vulcan.api.server:app --host 0.0.0.0 --port 8000',
      interpreter: 'none',
      env: {
        PYTHONPATH: '.',
      },
      watch: ['vulcan'],
      ignore_watch: ['*.log', '*.pyc', '__pycache__'],
      max_memory_restart: '2G',
      error_file: '../logs/api-error.log',
      out_file: '../logs/api-out.log',
      log_file: '../logs/api-combined.log',
      time: true
    },
    {
      name: 'vulcan-dashboard',
      cwd: './frontend',
      script: 'npm',
      args: 'run dev',
      interpreter: 'none',
      watch: false,
      max_memory_restart: '1G',
      error_file: '../logs/dashboard-error.log',
      out_file: '../logs/dashboard-out.log',
      log_file: '../logs/dashboard-combined.log',
      time: true
    }
  ]
};
EOF

# Create logs directory
mkdir -p logs

# Activate virtual environment if available
if [ -f ".venv/bin/activate" ]; then
    print_status "Activating Python virtual environment..."
    source .venv/bin/activate
elif [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
fi

# Stop any existing VULCAN processes
print_status "Stopping any existing VULCAN processes..."
pm2 stop vulcan-api vulcan-dashboard 2>/dev/null || true
pm2 delete vulcan-api vulcan-dashboard 2>/dev/null || true

# Kill processes on ports just in case
lsof -ti:8000 | xargs kill -9 2>/dev/null || true
lsof -ti:3000 | xargs kill -9 2>/dev/null || true

# Start services with PM2
print_status "Starting VULCAN services with PM2..."
pm2 start ecosystem.config.js

# Wait for services to start
print_status "Waiting for services to start..."
sleep 5

# Check service status
pm2 status

# Save PM2 configuration
pm2 save

# Setup PM2 startup script (optional)
print_status "To auto-start VULCAN on system boot, run:"
echo "  pm2 startup"
echo "  pm2 save"

# Print success message
echo -e "\n${GREEN}════════════════════════════════════════════${NC}"
print_success "VULCAN is running with PM2!"
echo -e "${GREEN}════════════════════════════════════════════${NC}\n"

print_status "API Server:   http://localhost:8000"
print_status "API Docs:     http://localhost:8000/docs"
print_status "Dashboard:    http://localhost:3000"

echo -e "\n${YELLOW}Useful PM2 commands:${NC}"
echo "  pm2 status         - Check service status"
echo "  pm2 logs           - View all logs"
echo "  pm2 logs vulcan-api    - View API logs"
echo "  pm2 logs vulcan-dashboard - View dashboard logs"
echo "  pm2 restart all    - Restart all services"
echo "  pm2 stop all       - Stop all services"
echo "  pm2 monit          - Real-time monitoring"

echo -e "\n${YELLOW}To stop VULCAN:${NC} pm2 stop all\n" 