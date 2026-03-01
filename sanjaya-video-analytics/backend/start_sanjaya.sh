#!/bin/bash

echo "════════════════════════════════════════════════════════════════"
echo "🚀 SANJAYA - Starting Services"
echo "════════════════════════════════════════════════════════════════"

# 1. Check if Neo4j container exists
if docker ps -a --format '{{.Names}}' | grep -q '^neo4j$'; then
    echo "📊 Neo4j container found"
    
    # Start if stopped
    if ! docker ps --format '{{.Names}}' | grep -q '^neo4j$'; then
        echo "🔄 Starting Neo4j container..."
        docker start neo4j
        sleep 5
    else
        echo "✅ Neo4j already running"
    fi
else
    echo "🆕 Creating new Neo4j container..."
    docker run -d \
        --name neo4j \
        -p 7474:7474 \
        -p 7687:7687 \
        -e NEO4J_AUTH=neo4j/neo4j123 \
        neo4j:5.15-community
    
    echo "⏳ Waiting for Neo4j to start (15s)..."
    sleep 15
fi

# 2. Verify Neo4j is responding
echo "🔍 Checking Neo4j connection..."
max_attempts=10
attempt=0

while [ $attempt -lt $max_attempts ]; do
    if curl -s http://localhost:7474 > /dev/null 2>&1; then
        echo "✅ Neo4j is ready!"
        break
    fi
    attempt=$((attempt + 1))
    echo "   Attempt $attempt/$max_attempts..."
    sleep 2
done

if [ $attempt -eq $max_attempts ]; then
    echo "❌ Neo4j failed to start after $max_attempts attempts"
    exit 1
fi

# 3. Activate conda and start Flask
echo ""
echo "🐍 Activating conda environment..."
source /home/admin-/miniconda3/etc/profile.d/conda.sh
conda activate sanjaya

echo ""
echo "🌐 Starting Flask app..."
echo "════════════════════════════════════════════════════════════════"
echo "📹 Upload videos at: http://localhost:5000"
echo "📊 Neo4j Browser: http://localhost:7474 (neo4j / neo4j123)"
echo "════════════════════════════════════════════════════════════════"
echo ""

python app.py
