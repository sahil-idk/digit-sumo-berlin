from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import sqlite3
from typing import List, Dict
from datetime import datetime

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

def get_db():
    conn = sqlite3.connect('traffic_simulation.db')
    conn.row_factory = sqlite3.Row
    return conn

@app.get("/simulation/traffic")
async def get_traffic_state():
    db = get_db()
    cursor = db.cursor()
    try:
        # Get latest simulation stats
        cursor.execute('''
            SELECT timestamp, vehicle_count, current_phase 
            FROM simulation_stats 
            ORDER BY timestamp DESC LIMIT 1
        ''')
        stats = cursor.fetchone()

        # Get current traffic light states
        cursor.execute('''
            SELECT traffic_light_id, state, phase 
            FROM traffic_light_states 
            WHERE timestamp = (SELECT MAX(timestamp) FROM traffic_light_states)
        ''')
        lights = cursor.fetchall()

        # Get current vehicle positions
        cursor.execute('''
            SELECT vehicle_id, edge_id, speed, waiting_time 
            FROM vehicle_stats 
            WHERE timestamp = (SELECT MAX(timestamp) FROM vehicle_stats)
        ''')
        vehicles = cursor.fetchall()

        return {
            "timestamp": stats["timestamp"] if stats else None,
            "vehicle_count": stats["vehicle_count"] if stats else 0,
            "current_phase": stats["current_phase"] if stats else None,
            "traffic_lights": [dict(row) for row in lights],
            "vehicles": [dict(row) for row in vehicles]
        }
    except sqlite3.Error as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()

@app.get("/simulation/history")
async def get_history(minutes: int = 5):
    db = get_db()
    cursor = db.cursor()
    try:
        cursor.execute('''
            SELECT timestamp, vehicle_count, current_phase 
            FROM simulation_stats 
            WHERE timestamp >= datetime('now', ?) 
            ORDER BY timestamp
        ''', (f'-{minutes} minutes',))
        return [dict(row) for row in cursor.fetchall()]
    finally:
        db.close()

@app.get("/traffic-lights/{light_id}")
async def get_traffic_light(light_id: str):
    db = get_db()
    cursor = db.cursor()
    try:
        cursor.execute('''
            SELECT * FROM traffic_light_states 
            WHERE traffic_light_id = ? 
            ORDER BY timestamp DESC LIMIT 100
        ''', (light_id,))
        return [dict(row) for row in cursor.fetchall()]
    finally:
        db.close()

@app.get("/vehicles/active")
async def get_active_vehicles():
    db = get_db()
    cursor = db.cursor()
    try:
        cursor.execute('''
            SELECT DISTINCT vehicle_id, edge_id, speed 
            FROM vehicle_stats 
            WHERE timestamp = (SELECT MAX(timestamp) FROM vehicle_stats)
        ''')
        return [dict(row) for row in cursor.fetchall()]
    finally:
        db.close()