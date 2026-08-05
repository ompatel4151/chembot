import os

# Bind to the port Render (or any host) provides.
bind = f"0.0.0.0:{os.environ.get('PORT', '10000')}"

# One worker keeps memory low for the free tier.
workers = 1
threads = 2

# The first request loads the embedding model, so allow extra startup time.
timeout = 120
