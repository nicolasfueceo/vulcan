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
